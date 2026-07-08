"""Optuna hyperparameter tuning ON TOP OF the existing leak-free inner-LOSO
DETECTION protocol (ADR-003/013/018). Task = DETECTION.

Reuses (does not reimplement) run_loso.py's run_split() and inner_val_score()
and basic_tcn.py's receptive_field(). Only NEW logic here is the Optuna search
space / objective / study bookkeeping.

CRITICAL ANTI-LEAKAGE RULE: the objective for a trial is the MEAN of the
per-outer-fold median inner-LOSO validation score (ROC-AUC, ADR-013's
original inner_val_score; or PR-AUC, ADR-018's inner_val_score_prauc --
selectable via make_objective's score_fn param, same median-selection RULE
either way) over a subset of outer folds. TEST subjects' data is NEVER
loaded during tuning -- build_xy() is only ever called with train_subjs / a
single inner-candidate val_subj, never with a test_subj. The outer
`test_subj` variable is used solely to define the pool of eligible inner
candidates and for the leakage assertion string; its data never enters
run_split().

After a study completes, the best params can be promoted to ONE final full
inner-LOSO run (n_epochs=50, all 8 outer folds) via run_finalize(), which
calls run_loso.main() -- the same production entry point, reused not
reimplemented -- so that run's test metrics are the first time the tuned
model ever sees a test subject.

Usage:
    python src/tune_optuna.py             # cheap ROC-AUC smoke only (ADR-013)
    python src/tune_optuna.py --full       # launch the full PR-AUC study (ADR-018):
                                            # 40 trials, tuning_folds=[S01,S03,S08],
                                            # epoch_cap=25, n_startup_trials=8
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from basic_tcn import BasicTCN, receptive_field  # noqa: E402  (RF guard reuse)
from run_loso import (  # noqa: E402  (reuse, do not reimplement -- ADR-013)
    causal_median,
    discover_subject_files,
    get_fog_subjects,
    inner_val_score,
    run_split,
)
import run_loso  # noqa: E402  (for run_loso.main(), the finalize step)


def inner_val_score_prauc(va_probs, va_labels):
    """PR-AUC analogue of run_loso.inner_val_score() (ADR-003's ROC-AUC-based
    median rule) -- ADR-018. Used ONLY as an alternative Optuna tuning
    objective; does NOT change run_loso.py's own val_subj selection rule,
    which stays ROC-AUC-based (ADR-003) and UNCHANGED for the production
    DETECTION pipeline. Same causal-median(k=5) smoothing as inner_val_score,
    same median-selection-rule usage pattern (see make_objective)."""
    if len(np.unique(va_labels)) < 2:
        return float("nan")
    va_smoothed = causal_median(va_probs, k=5)
    return average_precision_score(va_labels, va_smoothed)

OPTUNA_DIR = ROOT_DIR / "results" / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 64

# --- Search space: kernel_size/dilations restricted to the 4 RF<=64 combos
# (ADR-013). Optuna picks an INDEX into this list -- it can never construct
# an invalid (kernel_size, dilations) pair, so there is nothing to skip/prune
# for RF on the architecture axis (structural guarantee, not just a runtime
# check). RF is still recomputed and asserted per trial as a defensive guard
# (S5 in CLAUDE.md: verify RF before building any TCN).
ARCH_COMBOS = [
    {"kernel_size": 3, "dilations": [1, 2, 4]},
    {"kernel_size": 3, "dilations": [1, 4, 8]},
    {"kernel_size": 3, "dilations": [1, 2, 8]},
    {"kernel_size": 5, "dilations": [1, 2, 4]},
]


def _validate_arch_combos():
    """Static sanity check at import time: every ARCH_COMBOS entry must pass
    the RF<=window_size guard, using the SAME receptive_field() basic_tcn.py
    uses for real. Also demonstrates the guard REJECTS a known-bad combo
    (K=5, dilations=[1,4,8], RF=105) via BasicTCN's own ValueError."""
    rows = []
    for combo in ARCH_COMBOS:
        rf = receptive_field(combo["kernel_size"], combo["dilations"], n_conv_per_block=2)
        assert rf <= WINDOW_SIZE, f"ARCH_COMBOS entry {combo} has RF={rf} > {WINDOW_SIZE}!"
        rows.append({"kernel_size": combo["kernel_size"], "dilations": combo["dilations"], "RF": rf})

    bad_kernel_size, bad_dilations = 5, [1, 4, 8]
    bad_rf = receptive_field(bad_kernel_size, bad_dilations, n_conv_per_block=2)
    assert bad_rf > WINDOW_SIZE, "expected the known-bad combo to exceed window_size"
    rejected = False
    try:
        BasicTCN(n_channels=3, n_classes=1, kernel_size=bad_kernel_size,
                 dilations=bad_dilations, window_size=WINDOW_SIZE)
    except ValueError:
        rejected = True
    assert rejected, (
        f"RF GUARD FAILED TO REJECT invalid combo K={bad_kernel_size} "
        f"dilations={bad_dilations} (RF={bad_rf} > {WINDOW_SIZE})!"
    )
    return rows, bad_rf


def _append_row(csv_path, row):
    """Crash-safe incremental append, same pattern as run_loso.py's
    append_result_row(): one row per call, header only on first write."""
    row_df = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", header=write_header, index=False)


def make_objective(subject_files, fog_subjects, tuning_test_subjects, device, epoch_cap,
                    trial_log_path, split_timings, score_fn=None, metric_name="rocauc"):
    """Build the Optuna objective. VAL-ONLY: never loads a test_subj's data.

    score_fn: the per-inner-candidate scoring function, ADR-003's median rule
    applies to WHATEVER score_fn returns (rank the pool, pick the median).
    Default (None) -> inner_val_score (ROC-AUC, ADR-013). Pass
    inner_val_score_prauc for the PR-AUC objective (ADR-018).
    metric_name: used only for CSV column names / print labels, so the two
    objectives' logs are distinguishable (fold_median_val_<metric_name>)."""
    score_fn = score_fn or inner_val_score

    def objective(trial):
        arch_idx = trial.suggest_categorical("arch_idx", list(range(len(ARCH_COMBOS))))
        arch = ARCH_COMBOS[arch_idx]
        kernel_size, dilations = arch["kernel_size"], arch["dilations"]
        rf = receptive_field(kernel_size, dilations, n_conv_per_block=2)
        if rf > WINDOW_SIZE:
            # Unreachable given ARCH_COMBOS, kept as a loud defensive guard.
            raise optuna.TrialPruned(f"RF={rf} > {WINDOW_SIZE} for combo {arch}")

        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        alpha = trial.suggest_float("focal_alpha", 0.5, 0.9)
        gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [128, 256])

        fold_scores = []
        for step, test_subj in enumerate(tuning_test_subjects):
            # test_subj only defines the eligible candidate pool + leakage
            # assertion below -- its data is NEVER loaded (no build_xy call
            # with test_subj anywhere in this function).
            pool = [s for s in fog_subjects if s != test_subj]
            inner_scores = {}
            for candidate in pool:
                inner_train_subjs = [s for s in pool if s != candidate]
                assert test_subj not in inner_train_subjs and candidate != test_subj, \
                    f"LEAKAGE: test_subj {test_subj} touched by inner split (candidate={candidate})!"
                _, _, _, va_probs, va_labels, elapsed, epochs_run, ckpt_path = run_split(
                    inner_train_subjs, candidate, test_subj, subject_files, device, epoch_cap,
                    window_size=WINDOW_SIZE, kernel_size=kernel_size, dilations=dilations,
                    dropout=dropout, alpha=alpha, gamma=gamma, lr=lr, weight_decay=weight_decay,
                    batch_size=batch_size, split_timings=split_timings, verbose=False)
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)  # tuning never needs to keep a checkpoint
                inner_scores[candidate] = score_fn(va_probs, va_labels)

            valid_scores = {k: v for k, v in inner_scores.items() if not np.isnan(v)}
            if not valid_scores:
                raise optuna.TrialPruned(
                    f"All inner val candidates single-class for test={test_subj}")
            ranked = sorted(valid_scores.items(), key=lambda kv: kv[1])
            val_subj, median_score = ranked[len(ranked) // 2]
            fold_scores.append(median_score)

            running_mean = float(np.mean(fold_scores))
            _append_row(trial_log_path, {
                "trial_number": trial.number, "step": step, "test_subj_excluded_from_train": test_subj,
                "selected_val_subj": val_subj, f"fold_median_val_{metric_name}": round(median_score, 4),
                f"running_mean_val_{metric_name}": round(running_mean, 4),
                "arch_idx": arch_idx, "kernel_size": kernel_size, "dilations": str(dilations),
                "dropout": round(dropout, 4), "focal_alpha": round(alpha, 4),
                "focal_gamma": round(gamma, 4), "lr": lr, "weight_decay": weight_decay,
                "batch_size": batch_size,
            })

            trial.report(running_mean, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def _build_study(study_name, seed, n_startup_trials=5, n_warmup_steps=1):
    storage_path = OPTUNA_DIR / f"{study_name}.db"
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    return optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )


def _print_wallclock_projection(split_timings, n_trials_observed, folds_observed, epoch_cap_observed,
                                 n_trials_full=30, folds_full=3, epoch_cap_full=25):
    if not split_timings:
        print("No models trained -- cannot project wall-clock.")
        return
    total_elapsed = sum(e for e, _ in split_timings)
    total_epochs = sum(ep for _, ep in split_timings)
    n_models = len(split_timings)
    avg_time_per_epoch = total_elapsed / total_epochs if total_epochs else float("nan")

    inner_per_fold = 7  # tuning objective only trains inner candidates, no final retrain
    full_models = n_trials_full * folds_full * inner_per_fold
    upper_bound_s = avg_time_per_epoch * epoch_cap_full * full_models

    print("\n" + "=" * 60)
    print(" WALL-CLOCK PROJECTION for the FULL Optuna study")
    print("=" * 60)
    print(f"Observed this run: {n_models} model(s) trained "
          f"({n_trials_observed} trial(s) x up to {folds_observed} fold(s) x {inner_per_fold} inner candidates), "
          f"epoch cap={epoch_cap_observed}, total wall-clock={total_elapsed:.1f}s")
    print(f"Avg wall-clock per epoch: {avg_time_per_epoch:.2f}s")
    print(f"Projected FULL STUDY ({n_trials_full} trials x {folds_full} outer folds x "
          f"{inner_per_fold} inner candidates x {epoch_cap_full} epochs, upper bound / "
          f"no pruning benefit, no early stopping):")
    print(f"  {upper_bound_s:.0f}s  (~{upper_bound_s/60:.1f} min, ~{upper_bound_s/3600:.2f} hr)")
    print("This is a conservative UPPER BOUND: it assumes every trial runs every fold to")
    print("completion at the full epoch cap. In practice patience=5 early stopping AND the")
    print("MedianPruner (which can abort a clearly-bad trial after fold 1 of 3) will both")
    print("cut real wall-clock below this number, likely substantially once >=5 trials have")
    print("completed and the pruner has a real median to compare against.")

    # Lighter-weight alternative, purely informational (not a recommendation to change
    # the stated defaults, just cost context given how large the upper bound is).
    light_models = 15 * 1 * inner_per_fold
    light_s = avg_time_per_epoch * epoch_cap_full * light_models
    print(f"\nFor reference, a lighter config (15 trials x 1 outer fold x {epoch_cap_full} "
          f"epochs) upper bound: {light_s:.0f}s (~{light_s/60:.1f} min, ~{light_s/3600:.2f} hr).")


def run_smoke(seed=42, tuning_test_subjects=None, score_fn=None, metric_name="rocauc",
              study_name="study_smoke", epoch_cap=5, n_trials=2, n_startup_trials=5,
              n_trials_full=30, folds_full=3, epoch_cap_full=25):
    """2 trials x 1 outer fold x 5-epoch cap (defaults). Proves: RF filter
    rejects an invalid combo, the objective is val-only (test_subj data
    never loaded), trials persist (sqlite + CSV), no leakage assertion
    fires. Prints a wall-clock projection for the full study. Does NOT
    launch it.

    score_fn/metric_name: default (None) reproduces the original ADR-013
    ROC-AUC smoke exactly. Pass inner_val_score_prauc/"prauc" for the
    ADR-018 PR-AUC objective smoke (still the SAME harness -- median rule,
    leakage assertion, RF guard -- only the scoring function differs)."""
    score_fn = score_fn or inner_val_score
    print("=" * 60)
    print(f" OPTUNA SMOKE: {n_trials} trials x {len(tuning_test_subjects) if tuning_test_subjects else 1} "
          f"outer fold(s) x {epoch_cap}-epoch cap, objective=val_{metric_name}")
    print("=" * 60)

    print("\n--- RF guard validation ---")
    rows, bad_rf = _validate_arch_combos()
    for r in rows:
        print(f"  ARCH OK: kernel_size={r['kernel_size']} dilations={r['dilations']} RF={r['RF']} (<= {WINDOW_SIZE})")
    print(f"  REJECTED (expected): kernel_size=5 dilations=[1, 4, 8] RF={bad_rf} (> {WINDOW_SIZE}) "
          f"-- BasicTCN raised ValueError as designed.")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[DEVICE] {device}")

    subject_files = discover_subject_files(ROOT_DIR)
    fog_subjects = get_fog_subjects(subject_files)
    tuning_test_subjects = tuning_test_subjects if tuning_test_subjects is not None else fog_subjects[:1]
    print(f"FOG_SUBJECTS: {fog_subjects}")
    print(f"Tuning outer-fold subset (test subjects EXCLUDED from all training/val this call): "
          f"{tuning_test_subjects}")
    print("VAL-ONLY GUARD: build_xy()/run_split() are only ever called with train_subjs or a "
          "single inner-candidate val_subj -- the leakage assertion "
          "'test_subj not in inner_train_subjs and candidate != test_subj' fires on EVERY inner "
          "candidate of every fold, every trial (see make_objective). A completed run with no "
          "AssertionError raised is a live confirmation this held for the entire smoke.")

    trial_log_path = OPTUNA_DIR / f"trials_log_{study_name[len('study_'):]}.csv"
    if trial_log_path.exists():
        os.remove(trial_log_path)  # smoke is meant to be re-runnable / disposable

    split_timings = []
    objective = make_objective(subject_files, fog_subjects, tuning_test_subjects, device,
                                epoch_cap=epoch_cap, trial_log_path=trial_log_path,
                                split_timings=split_timings, score_fn=score_fn, metric_name=metric_name)

    study = _build_study(study_name, seed=seed, n_startup_trials=n_startup_trials)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - t0

    print(f"\nSmoke study done in {elapsed:.1f}s. Trials:")
    for t in study.trials:
        print(f"  trial {t.number}: state={t.state.name} value={t.value} params={t.params}")
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"\n{n_complete}/{len(study.trials)} trials COMPLETE (no AssertionError raised -> "
          f"leakage guard held for every inner candidate of every fold of every trial).")
    print(f"Best value (val-only mean inner {metric_name.upper()}): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"\nStudy persisted to: {OPTUNA_DIR / (study_name + '.db')} (sqlite)")
    print(f"Per-trial-per-fold log persisted to: {trial_log_path}")

    _print_wallclock_projection(split_timings, n_trials_observed=n_trials,
                                 folds_observed=(len(tuning_test_subjects) if tuning_test_subjects else 1),
                                 epoch_cap_observed=epoch_cap, n_trials_full=n_trials_full,
                                 folds_full=folds_full, epoch_cap_full=epoch_cap_full)
    return study


def run_full_study(n_trials=30, epoch_cap=25, tuning_folds=None, seed=42,
                    score_fn=None, metric_name="rocauc", n_startup_trials=5, study_name="study_full"):
    """NOT called automatically by anything in this module (except the
    --full CLI path, see __main__). Launch only after reviewing a smoke's
    wall-clock projection and approving the time cost."""
    score_fn = score_fn or inner_val_score
    print("=" * 60)
    print(f" OPTUNA FULL STUDY: {n_trials} trials, epoch_cap={epoch_cap}, objective=val_{metric_name}, "
          f"n_startup_trials={n_startup_trials}")
    print("=" * 60)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    subject_files = discover_subject_files(ROOT_DIR)
    fog_subjects = get_fog_subjects(subject_files)
    tuning_test_subjects = tuning_folds if tuning_folds is not None else fog_subjects[:3]
    print(f"FOG_SUBJECTS: {fog_subjects}")
    print(f"Tuning outer-fold subset (REDUCED protocol, test subjects held out from tuning "
          f"entirely): {tuning_test_subjects}")
    print("VAL-ONLY GUARD: same leakage assertion as the smoke (make_objective), asserted on "
          "every inner candidate of every fold of every trial -- an AssertionError would abort "
          "this run immediately rather than silently leak.")

    trial_log_path = OPTUNA_DIR / f"trials_log_{study_name[len('study_'):]}.csv"
    split_timings = []
    objective = make_objective(subject_files, fog_subjects, tuning_test_subjects, device,
                                epoch_cap=epoch_cap, trial_log_path=trial_log_path,
                                split_timings=split_timings, score_fn=score_fn, metric_name=metric_name)

    study = _build_study(study_name, seed=seed, n_startup_trials=n_startup_trials)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - t0

    print(f"\nFull study done in {elapsed:.1f}s (~{elapsed/3600:.2f} hr).")
    print(f"Best value (val-only mean inner {metric_name.upper()}): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Study persisted to: {OPTUNA_DIR / (study_name + '.db')} (sqlite)")
    print(f"Per-trial-per-fold log persisted to: {trial_log_path}")
    return study


def run_finalize(study_name="study_full", n_epochs=50, seed=42):
    """Load best params from a completed study and run ONE final full
    inner-LOSO (all 8 outer folds) via run_loso.main() -- reused, not
    reimplemented. This is the first time the tuned model sees any test
    subject's data."""
    storage_path = OPTUNA_DIR / f"{study_name}.db"
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{storage_path}")
    best = study.best_params
    arch = ARCH_COMBOS[best["arch_idx"]]
    print(f"Loaded study '{study_name}': best_value={study.best_value:.4f}, best_params={best}")
    print(f"Resolved architecture: kernel_size={arch['kernel_size']}, dilations={arch['dilations']}")

    return run_loso.main(
        n_epochs=n_epochs, subjects_to_run=None, seed=seed,
        kernel_size=arch["kernel_size"], dilations=arch["dilations"],
        dropout=best["dropout"], alpha=best["focal_alpha"], gamma=best["focal_gamma"],
        lr=best["lr"], weight_decay=best["weight_decay"], batch_size=best["batch_size"],
    )


# ADR-018 config: the overnight PR-AUC DETECTION study. Kept as named
# constants (not buried in argparse defaults) so DECISIONS.md/PROJECT_STATUS.md
# can reference the exact same values the launch actually used.
PRAUC_TUNING_FOLDS = ["S01", "S03", "S08"]  # deliberately includes S08 (akinetic outlier)
PRAUC_N_TRIALS = 40
PRAUC_EPOCH_CAP = 25
PRAUC_N_STARTUP_TRIALS = 8
PRAUC_SEED = 42
PRAUC_FULL_STUDY_NAME = "study_full_prauc"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Launch the FULL PR-AUC study (ADR-018): "
                              f"{PRAUC_N_TRIALS} trials, tuning_folds={PRAUC_TUNING_FOLDS}, "
                              f"epoch_cap={PRAUC_EPOCH_CAP}, n_startup_trials={PRAUC_N_STARTUP_TRIALS}. "
                              "Default (no flag): run only the cheap original ROC-AUC smoke (ADR-013).")
    args = parser.parse_args()

    if args.full:
        run_full_study(
            n_trials=PRAUC_N_TRIALS, epoch_cap=PRAUC_EPOCH_CAP, tuning_folds=PRAUC_TUNING_FOLDS,
            seed=PRAUC_SEED, score_fn=inner_val_score_prauc, metric_name="prauc",
            n_startup_trials=PRAUC_N_STARTUP_TRIALS, study_name=PRAUC_FULL_STUDY_NAME,
        )
    else:
        # Safety rail: running this file directly with no flags only ever does
        # the cheap ROC-AUC smoke (ADR-013's original, unchanged for
        # reproducibility). The PR-AUC smoke, full study, and finalize step
        # are explicit function calls / the --full flag, never auto-triggered.
        run_smoke()
