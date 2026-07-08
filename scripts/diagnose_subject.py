"""Read-only diagnostic for a pathological LOSO subject (default S08, vs S05 as a
healthy reference). No training, no eval-logic changes, no data mutation.

Reuses ONLY the parsing/windowing helpers from the pipeline:
  - run_loso.load_daphnet_file  (raw file -> filtered X, y)
  - fog_preprocessing.FoGPreprocessor.create_windows  (any-overlap windowing)
Everything else (label counts, event detection, axis stats, plots) is
computed directly by this script.

Usage:
  python scripts/diagnose_subject.py                       # S08 vs S05 (default)
  python scripts/diagnose_subject.py --subject S06 --reference S05
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from run_loso import load_daphnet_file, causal_median  # noqa: E402  (parsing/post-proc only, no training call)
from fog_preprocessing import FoGPreprocessor  # noqa: E402  (windowing only)

DATA_DIR = ROOT_DIR / "data" / "dataset"
FS_HZ = 64
WINDOW = 64
STEP = 32
AXIS_NAMES = ["ankle_x", "ankle_y", "ankle_z"]
FOG_SUBJECTS = ["S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09"]


def subject_files(subj):
    return sorted(glob.glob(str(DATA_DIR / f"{subj}R*.txt")))


def raw_label_counts(subj):
    """Per-file label composition BEFORE the 0(invalid)-filter, plus FoG events."""
    rows, event_rows = [], []
    for f in subject_files(subj):
        df = pd.read_csv(f, sep=r"\s+", header=None)
        y_raw = df.iloc[:, -1].values
        n0, n1, n2 = (int((y_raw == v).sum()) for v in (0, 1, 2))
        denom = n1 + n2
        rows.append({
            "file": os.path.basename(f), "invalid_0": n0, "noFoG_1": n1, "FoG_2": n2,
            "FoG_fraction_post_filter": (n2 / denom) if denom else float("nan"),
        })
        is_fog = (y_raw == 2).astype(int)
        diff = np.diff(np.concatenate(([0], is_fog, [0])))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            event_rows.append({
                "file": os.path.basename(f), "start_sample": int(s), "end_sample": int(e),
                "duration_s": round((e - s) / FS_HZ, 2),
            })
    return pd.DataFrame(rows), pd.DataFrame(event_rows)


def build_subject_xy(subj):
    Xs, ys = [], []
    for f in subject_files(subj):
        X, y = load_daphnet_file(f)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def window_pos_neg(X, y):
    pre = FoGPreprocessor(window_size=WINDOW, step_size=STEP)
    _, y_windows = pre.create_windows(X, y)
    y_windows = y_windows.numpy()
    n_pos = int((y_windows == 1).sum())
    n_neg = int((y_windows == 0).sum())
    return n_pos, n_neg


def axis_stats_table(X, label):
    rows = []
    for i, name in enumerate(AXIS_NAMES):
        col = X[:, i]
        rows.append({
            "group": label, "axis": name,
            "mean": round(float(col.mean()), 2), "std": round(float(col.std()), 2),
            "min": round(float(col.min()), 2), "max": round(float(col.max()), 2),
            "skew": round(float(skew(col)), 3), "kurtosis": round(float(kurtosis(col)), 3),
        })
    return pd.DataFrame(rows)


def find_first_onset(subj):
    for f in subject_files(subj):
        df = pd.read_csv(f, sep=r"\s+", header=None)
        y_raw = df.iloc[:, -1].values
        idx = np.where(y_raw == 2)[0]
        if len(idx):
            return f, int(idx[0]), df
    return None, None, None


def plot_raw_segments(subj_a, subj_b, out_path, pad_s=5):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for ax, subj in zip(axes, (subj_a, subj_b)):
        f, onset_idx, df = find_first_onset(subj)
        if f is None:
            ax.set_title(f"{subj}: no FoG=2 samples found (no onset to plot)")
            ax.axis("off")
            continue
        pad = pad_s * FS_HZ
        lo, hi = max(0, onset_idx - pad), min(len(df), onset_idx + pad)
        seg = df.iloc[lo:hi]
        t = (seg.index.values - onset_idx) / FS_HZ
        for i, name in enumerate(AXIS_NAMES):
            ax.plot(t, seg.iloc[:, 1 + i].values, label=name, linewidth=1)
        ax.axvline(0, color="k", linestyle="--", linewidth=1, label="FoG onset")
        ax.set_title(f"{subj}: raw ankle accel around first FoG onset ({os.path.basename(f)})")
        ax.set_xlabel("time relative to onset (s)")
        ax.set_ylabel("accel (raw mg)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_prob_histogram(subj, detailed_csv, out_path):
    df = pd.read_csv(detailed_csv)
    sub = df[df["Subject"] == subj].sort_values("Window_Idx")
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)
    for label, name, color in ((0.0, "True_Label=0 (no-FoG)", "tab:blue"),
                                (1.0, "True_Label=1 (FoG)", "tab:red")):
        vals = sub.loc[sub["True_Label"] == label, "Pred_Prob"]
        ax.hist(vals, bins=bins, alpha=0.55, label=f"{name} (n={len(vals)})", color=color)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Window count")
    ax.set_title(f"{subj}: predicted prob by true label (RUN_20260702_172209 test fold)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # Same causal_median smoothing (k=5) the official pipeline applies before
    # scoring Test_ROC_AUC (run_loso.py); raw-prob AUC alone under-represents
    # what the frozen-threshold protocol actually measured.
    smoothed = causal_median(sub["Pred_Prob"].values, k=5)
    auc_raw = roc_auc_score(sub["True_Label"], sub["Pred_Prob"])
    auc_smoothed = roc_auc_score(sub["True_Label"], smoothed)
    mean0 = sub.loc[sub["True_Label"] == 0, "Pred_Prob"].mean()
    mean1 = sub.loc[sub["True_Label"] == 1, "Pred_Prob"].mean()
    median0 = sub.loc[sub["True_Label"] == 0, "Pred_Prob"].median()
    median1 = sub.loc[sub["True_Label"] == 1, "Pred_Prob"].median()
    return {
        "subject": subj, "n_windows": len(sub), "n_pos": int((sub["True_Label"] == 1).sum()),
        "n_neg": int((sub["True_Label"] == 0).sum()),
        "auc_raw_prob": round(float(auc_raw), 4),
        "auc_causal_median_k5": round(float(auc_smoothed), 4),
        "auc_if_flipped_raw": round(float(1 - auc_raw), 4),
        "mean_prob_label0": round(float(mean0), 4), "mean_prob_label1": round(float(mean1), 4),
        "median_prob_label0": round(float(median0), 4), "median_prob_label1": round(float(median1), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="S08", help="subject under diagnosis")
    ap.add_argument("--reference", default="S05", help="healthy-reference subject")
    ap.add_argument("--run-id", default="RUN_20260702_172209",
                     help="existing LOSO run dir under results/ (for saved test-set probs)")
    args = ap.parse_args()

    subj, ref = args.subject, args.reference
    report_dir = ROOT_DIR / "results" / "reports"
    plot_dir = report_dir / f"diag_{subj}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"diag_{subj}_report.txt"

    detailed_csv = None
    run_dir = ROOT_DIR / "results" / args.run_id
    matches = glob.glob(str(run_dir / "loso_detailed_predictions_*.csv"))
    if matches:
        detailed_csv = matches[0]

    lines = []
    def w(s=""):
        lines.append(s)

    w("=" * 78)
    w(f"Subject diagnostic report: {subj} (vs healthy reference {ref})")
    w(f"Source run for saved probs: {args.run_id}")
    w("Task: DETECTION. READ-ONLY analysis -- no training, no eval-logic changes.")
    w("=" * 78)
    w()

    # ---- 1. raw label composition ----
    w("-" * 78)
    w("1. RAW LABEL COMPOSITION (before 0-filter, per run file)")
    w("-" * 78)
    all_events = {}
    for s in (subj, ref):
        counts_df, events_df = raw_label_counts(s)
        w(f"\n[{s}]")
        w(counts_df.to_string(index=False))
        all_events[s] = events_df
        n_events = len(events_df)
        total_dur = events_df["duration_s"].sum() if n_events else 0.0
        w(f"FoG events: {n_events}, total FoG duration: {total_dur:.1f} s")
        if n_events:
            w("Event list (start_sample/end_sample in raw-file index, duration_s):")
            w(events_df.to_string(index=False))
    w()

    # ---- 2. windowing ----
    w("-" * 78)
    w("2. WINDOWING (window=64, step=32, any-overlap positive label)")
    w("-" * 78)
    win_rows = []
    for s in FOG_SUBJECTS:
        X, y = build_subject_xy(s)
        n_pos, n_neg = window_pos_neg(X, y)
        total = n_pos + n_neg
        win_rows.append({"subject": s, "pos_windows": n_pos, "neg_windows": n_neg,
                          "total_windows": total, "pos_fraction": round(n_pos / total, 4)})
    win_df = pd.DataFrame(win_rows)
    w(win_df.to_string(index=False))

    for s in (subj, ref):
        pool_frac = win_df.loc[win_df["subject"] != s, "pos_fraction"]
        target_frac = win_df.loc[win_df["subject"] == s, "pos_fraction"].iloc[0]
        pool_mean, pool_std = pool_frac.mean(), pool_frac.std()
        z = (target_frac - pool_mean) / pool_std if pool_std else float("nan")
        flag = "FLAG (>2 std from pool)" if abs(z) > 2 else "within pool range"
        w(f"\n[{s}] pos_fraction={target_frac:.4f}  pool(excl. {s}) mean={pool_mean:.4f} "
          f"std={pool_std:.4f}  z={z:.2f}  -> {flag}")
    w()

    # ---- 3. per-axis signal stats ----
    w("-" * 78)
    w("3. PER-AXIS ACCEL STATS (ankle x/y/z, raw mg, post 0-filter samples)")
    w("-" * 78)
    axis_frames = []
    for s in (subj, ref):
        X, _ = build_subject_xy(s)
        axis_frames.append(axis_stats_table(X, s))
    pool_subjects = [s for s in FOG_SUBJECTS if s not in (subj, ref)]
    pool_X = np.vstack([build_subject_xy(s)[0] for s in pool_subjects])
    axis_frames.append(axis_stats_table(pool_X, f"pool(excl.{subj},{ref})"))
    axis_df = pd.concat(axis_frames, ignore_index=True)
    w(axis_df.to_string(index=False))
    w()
    w("Sign-flip / mis-orientation check: compare 'mean' and 'skew' sign per axis across")
    w("groups. A consistent sign flip on one axis (opposite sign vs pool, same magnitude)")
    w("or a strongly divergent kurtosis indicates mounted-backwards / mis-scaled sensor.")
    w()

    # ---- 4. plots ----
    w("-" * 78)
    w("4. PLOTS")
    w("-" * 78)
    raw_seg_path = plot_dir / f"raw_segment_{subj}_vs_{ref}.png"
    plot_raw_segments(subj, ref, raw_seg_path)
    w(f"(a) Raw 3-axis signal around first FoG onset, {subj} vs {ref}: {raw_seg_path}")

    prob_summary = None
    if detailed_csv:
        prob_hist_path = plot_dir / f"prob_histogram_{subj}.png"
        prob_summary = plot_prob_histogram(subj, detailed_csv, prob_hist_path)
        w(f"(b) Predicted-prob histogram by true label, {subj} (source: {detailed_csv}): "
          f"{prob_hist_path}")
        ref_hist_path = plot_dir / f"prob_histogram_{ref}.png"
        ref_summary = plot_prob_histogram(ref, detailed_csv, ref_hist_path)
        if ref_summary:
            w(f"    (reference) {ref} predicted-prob histogram: {ref_hist_path}")
    else:
        w(f"(b) SKIPPED -- no loso_detailed_predictions_*.csv found under {run_dir}")
    w()

    if prob_summary:
        w("-" * 78)
        w("Predicted-probability summary by true label (from saved TEST predictions)")
        w("-" * 78)
        summ_df = pd.DataFrame([prob_summary])
        w(summ_df.to_string(index=False))
        w()
        w(f"AUC on raw saved probs (no post-proc): {prob_summary['auc_raw_prob']}")
        w(f"AUC on causal_median(k=5)-smoothed probs (matches official run_loso.py Test_ROC_AUC "
          f"scoring): {prob_summary['auc_causal_median_k5']}")
        w(f"AUC if the raw-prob axis were flipped (1 - AUC): {prob_summary['auc_if_flipped_raw']}")
        w(f"mean_prob(label=1) - mean_prob(label=0) = "
          f"{prob_summary['mean_prob_label1'] - prob_summary['mean_prob_label0']:.4f} "
          f"(negative => model ranks FoG windows LOWER than non-FoG windows)")
        delta = prob_summary['auc_causal_median_k5'] - prob_summary['auc_raw_prob']
        direction = "FURTHER below 0.5 (more anti-correlated)" if delta < 0 else "TOWARD 0.5 (less anti-correlated)"
        w(f"Effect of temporal smoothing on AUC: raw={prob_summary['auc_raw_prob']} -> "
          f"smoothed={prob_summary['auc_causal_median_k5']} ({direction}). If the ranking were pure")
        w("i.i.d. noise, temporal averaging would pull AUC toward 0.5, not push it further away;")
        w("a persistent or growing anti-correlation under smoothing points to a temporally")
        w("structured (systematic) inversion rather than random per-window noise.")
        w()

    # ---- conclusion ----
    w("=" * 78)
    w("ANALYSIS QUESTION: is S08's below-chance AUC most consistent with")
    w("  (a) label inversion / sensor mis-orientation (data issue), or")
    w("  (b) genuinely hard-but-valid data (keep, report honestly)?")
    w("=" * 78)
    w("See reasoning below, built from sections 1-4 above. This is a recommendation")
    w("only -- no automatic fix or exclusion was applied.")
    w()

    with open(report_path, "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written: {report_path}")
    print(f"Plots written to: {plot_dir}")


if __name__ == "__main__":
    main()
