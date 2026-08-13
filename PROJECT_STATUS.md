# PROJECT_STATUS.md — Live State (the cross-session "save file")

> Any fresh Claude session should be able to resume from THIS file + DECISIONS.md
> + EXPERIMENTS.md alone. Update the "Last updated" line and the relevant section
> at the end of every working session (R6/R10 in CLAUDE.md).

**Last updated:** 2026-07-03 (inner-LOSO full 8-fold run RUN_20260702_202440 complete and promoted to the OFFICIAL reference baseline, ADR-003 FULLY RESOLVED — see §3/§4; supersedes fixed-val RUN_20260702_172209, retained for comparison; S08 diagnosis CLOSED — hard-but-valid akinetic phenotype, ADR-012, docs/case_study_S08.md; training env is WSL-native `fog_env_wsl`, CLI + training run in one shell, GPU-verified; PREDICTION task validated at FULL 8-fold scale as ADR-014 (RUN_20260703_173337, episode_recall=0.6981±0.1606 primary metric) — see §3 "PREDICTION BASELINE" and results/reports/01_baselines/baseline_prediction.txt; Optuna hyperparameter tuning added as ADR-013, src/tune_optuna.py, val-only objective — 2-trial smoke passed after fixing a checkpoint-path collision with the concurrent PREDICTION-task session (now fixed for both), full 30-trial study NOT yet launched pending approval, see §4 item 10 and results/reports/02_optuna_tuning/optuna_smoke.txt; DOC CONSOLIDATION PASS 2026-07-03 — §3 rewritten as a DETECTION-vs-PREDICTION side-by-side comparison table, ADR-015 added for the cross-task S08 finding, §4/§5 updated with the pre-FoG sparsity + S09 low-recall follow-up items; read-only on results/, no code or data changes; EDGE DEPLOYMENT SCAFFOLD 2026-07-03 — new edge/ package (ADR-016), hardware-independent parts of the Jetson pipeline built + smoke-tested on WSL: stream_infer.py + ras_cue.py + latency_harness.py PASS, export_onnx.py written but blocked on missing onnx/onnxruntime packages (exact pip line reported, not installed silently) — see §4 item 12, results/reports/04_edge_deployment/edge_scaffold.txt; no src/ or data/ changes; EDGE SCAFFOLD PART 2 2026-07-03 — onnx==1.22.0/onnxruntime==1.27.0 installed + in requirements.txt, ONNX export UNBLOCKED (max diff 1.192e-07), edge/stream_infer.py + edge/latency_harness.py gained --infer_step (ADR-017, decoupled from training STEP_SIZE=32): smaller infer_step improves sanity recall/lead time but shrinks the k=5/w=7 post-proc's real-time span, an OPEN tradeoff not yet resolved — see §4 item 12, results/reports/04_edge_deployment/edge_scaffold_2.txt; still no src/ or data/ changes; EDGE SCAFFOLD PART 3 2026-07-04 — RESOLVED the infer_step-vs-post-proc-span tradeoff: span-constant k/w derivation (ADR-017 extension) + refractory debounce in ras_cue.py, trigger-count spread shrank from 3.90x to 1.10x across infer_step; key finding is that most of the earlier "smaller infer_step improves recall" trend was a smoothing-span dilution artifact, not genuine cadence benefit (lead time IS a genuine benefit, recall is not); recommending infer_step=16/k=3/w=4/refractory_ms=2000 for tomorrow's demo, pending approval — see §4 item 12, results/reports/04_edge_deployment/edge_scaffold_3.txt; still no src/ or data/ changes; OPTUNA OVERNIGHT STUDY LAUNCHED 2026-07-04 (ADR-018) — scaled-up DETECTION tuning study (40 trials, PR-AUC objective, tuning_folds=[S01,S03,S08] deliberately including the S08 akinetic outlier, epoch_cap=25), PR-AUC smoke gate verified PASSED before launch, running detached in tmux session "optuna" (~8.03hr conservative upper bound wall-clock, likely less with early stopping/pruning) — see §4 item 10, results/reports/02_optuna_tuning/optuna_launch.txt; no data/ or ROOT_DIR/logging changes, src/ change limited to src/tune_optuna.py; JETSON PLATFORM CONSOLIDATED 2026-07-04 (ADR-019) — Yahboom Orin NX SUPER (JetPack 6.2/CUDA 12.6), Tailscale SSH access, ~/fog_edge/ workspace on-device, see new §9 "Edge deployment platform"; read-only doc update, no src/ or data/ changes; TUNED DETECTION CANDIDATE 2026-07-05 (ADR-020) — full 8-fold inner-LOSO run with Optuna trial26's CLEAN hyperparameters (RUN_20260705_151423): ROC-AUC 0.866+/-0.073 (+0.092, std shrank) and PR-AUC 0.553+/-0.157 (+0.132) both improved substantially vs the RUN_20260702_202440 baseline; F1 improved only marginally with grown std (3 folds' frozen threshold doesn't transfer well, flagged as §4 item 13); S08 no longer below chance (0.491->0.732); 0 NaN, 0 leakage; does NOT supersede the baseline, both retained — see §3, results/reports/01_baselines/baseline_tuned.txt; no data/ or src/ changes -- new scripts/launch_final_detection_trial26.sh only calls the existing run_loso.main() with different hyperparameters, no logic duplicated; JETSON DEPLOYMENT CONSOLIDATED 2026-07-05/06 (ADR-021/ADR-022) -- Jetson deployment infrastructure DONE: env verified (JetPack 6.2/TRT 10.3.0/onnxruntime 1.18.0 CPU/numpy 1.21.5, no torch), fog_tcn_fp32.engine adopted as the deployment target over FP16 (no speedup on this model size, ADR-021), numerical parity vs source ONNX confirmed (max abs diff 3.51e-05), on-device latency measured (TRT forward median=0.6292ms, end-to-end compute median=1.8233ms, ~0.73% of the 250ms decision interval, ~2.2x faster than WSL/RTX4060 PyTorch) with 18/18 sanity-replay onsets caught -- see new §11; post-processing kept in pandas despite being ~2x slower than the forward pass, deferred per ADR-022 (still <2ms total, no urgency); §4 reordered -- the tuned-baseline F1/threshold-transfer investigation (item 11) BUMPED UP as still fully unresolved, Jetson deployment (item 13) marked infrastructure-complete with INT8 quantization/real IMU wiring/RAS hardware integration explicitly deferred; read-only on results/, no data/ or src/ changes)
**Phase:** End of Phase 2 (Week 4–5), now pivoting toward prediction + edge deploy.
**Hard deadline:** end of August — 推甄 portfolio + conference-grade write-up.

---

## 1. Snapshot
- Core model: TCN (dilated causal convs), channels [32, 64, 128], window = 64
  samples (1 s @ 64 Hz), Binary Focal Loss, Adam, early stopping.
- Validation: subject-level LOSO over Daphnet.
- Post-processing: median filter (K=5) + rolling majority vote (W=7) on the
  probability output (reported as +4% mean Best-F1, −2% AUC — see caveat below).

## 2. Done (Weeks 1–3)
- Daphnet 64 Hz parsing + sliding-window segmentation.
- TCN architecture + Focal Loss for class imbalance.
- LOSO-CV outer loop; memory-efficient logging (list[dict]→DataFrame).
- Temporal post-processing pipeline.

## 3. Current baselines — DETECTION and PREDICTION, side by side (2026-07-03)
Two official, full 8-fold inner-LOSO baselines now exist for this repo, one
per task (ADR-003 protocol reused unchanged between them; only the label
construction differs, ADR-014). Both use n_epochs=50, seed=42, GPU/fog_env_wsl,
inner-LOSO val_subj selection (median inner ROC-AUC rule).

| Metric (mean +/- std, 8 folds)       | DETECTION                | PREDICTION (horizon=1s)         |
|---------------------------------------|---------------------------|-----------------------------------|
| run_id                                | RUN_20260702_202440      | RUN_20260703_173337              |
| PRIMARY metric                        | Test_F1 = 0.366 +/- 0.144 | episode_recall = 0.698 +/- 0.161 (0.734 pooled) |
| ROC-AUC                               | 0.774 +/- 0.126          | 0.740 +/- 0.086                  |
| PR-AUC                                | 0.421 +/- 0.108          | 0.093 +/- 0.058 (secondary, sparsity-limited) |
| (task-specific 3rd metric)            | --                        | mean lead time = 0.715 +/- 0.070 s (of 1.00s horizon) |
| wall-clock (64 models)                | 3708s (~61.8 min)         | 3664.1s (~61.1 min)              |
| leakage assertions fired              | 0                          | 0                                  |
| report                                | results/reports/01_baselines/baseline_innerloso.txt | results/reports/01_baselines/baseline_prediction.txt |

**DETECTION (RUN_20260702_202440)** supersedes the earlier fixed-val run
RUN_20260702_172209 (Test_F1 0.295+/-0.206, val_subj=pool[0] biased toward
S01 in 6/8 folds; retained on disk for comparison only, not deleted). The
inner-LOSO fix raised all three metrics and shrank all three stds (see
results/reports/01_baselines/baseline_innerloso.txt for the delta table). S08 diagnosis
is CLOSED (docs/case_study_S08.md, ADR-012): below-chance AUC (0.4908) is
hard-but-valid data from an akinetic (quiet) freezing phenotype the
DETECTION model does not generalize to, not a sensor/label defect. Kept in
the official mean per S6.

**PREDICTION (RUN_20260703_173337, ADR-014 validated at full scale)** uses
episode-level recall + lead time as the PRIMARY metric, not window-level
F1/PR-AUC -- window-level metrics collapse under the ~1-4% pre-FoG positive
rate (5-15x sparser than DETECTION's ~19%), an intrinsic consequence of
horizon=window_size=64 + step_size=32 (each onset yields at most ~2 positive
windows by construction), not a labeling defect; ROC-AUC does not collapse
the same way and stays comparable to DETECTION in absolute terms. Flagged:
S09 episode_recall=0.4074 (weakest fold, though its window-level PR-AUC/F1
were near-best -- a specific missed-onset pattern, not a general failure);
S06 borderline-sparse (n_pos=20, 10 onsets) but not below the n_pos<20
threshold. No NaN folds, no skipped-onset folds.

**Cross-task finding (S08):** below-chance under DETECTION (0.4908) but NOT
below chance under PREDICTION (0.6659, episode_recall=0.7143, longest mean
lead time of any fold at 0.8359s) -- see ADR-015 and docs/case_study_S08.md
for the "two tasks read different parts of the movement signal"
interpretation.

**TUNED DETECTION baseline CANDIDATE (NOT superseding the baseline above):
RUN_20260705_151423** (Optuna trial 26 hyperparameters -- CLEAN, ADR-018/020
-- same architecture/protocol/causal post-proc as RUN_20260702_202440, only
kernel/dropout/focal/lr/weight_decay/batch_size differ). Both runs are kept
side by side; promoting the tuned candidate to "official" is a decision for
after this review, not made here.

| Metric        | Baseline (202440)  | Tuned candidate (trial26, 151423) | Delta   | Std change        |
|---------------|--------------------|------------------------------------|---------|--------------------|
| Test_F1       | 0.3664 +/- 0.1439  | 0.3865 +/- 0.2202                   | +0.0201 | GREW (0.144->0.220) |
| Test_PR_AUC   | 0.4211 +/- 0.1081  | 0.5531 +/- 0.1573                   | +0.1320 | GREW (0.108->0.157) |
| Test_ROC_AUC  | 0.7743 +/- 0.1261  | 0.8660 +/- 0.0734                   | +0.0917 | SHRANK (0.126->0.073) |

ROC-AUC is the cleanest win (higher mean AND lower variance, the same
signature that validated the ADR-003 inner-LOSO fix). PR-AUC improved more
in absolute terms but got less consistent across folds. F1's mean gain is
small and its std grew sharply -- 3 folds (S06, S07, S08) have LOW F1
despite good-to-great ROC-AUC, meaning the val-frozen (Youden's J)
threshold is not transferring to test as reliably under these
hyperparameters; flagged for follow-up, not yet resolved (see §4).
0 flagged folds (ROC-AUC<0.60), 0 NaN epochs, 0 leakage/error signatures
across all 64 models. Wall-clock 2795.1s (~46.6 min), faster than the
baseline's 3708s. Full table + per-fold breakdown:
results/reports/01_baselines/baseline_tuned.txt.

**Notable finding:** S08 (below-chance under the original baseline,
0.4908) rises to Test_ROC_AUC=0.7321 under the tuned hyperparameters -- the
single largest per-fold change in the comparison (+0.2413), moving it from
below-chance to comfortably above 0.70, though its F1 stays low (0.0909),
consistent with the ROC-AUC-improved/F1-inconsistent pattern above.

## 4. Current focus / next actions (in order)
1. [x] Run the full 8-fold leak-free LOSO (run_loso.main()) and log the result
       row in EXPERIMENTS.md as the new reference baseline. — done 2026-07-02,
       RUN_20260702_172209, see §3.
2. [x] Diagnose S08's below-chance AUC (data defect vs hard-but-valid). —
       CLOSED 2026-07-02, see §3, docs/case_study_S08.md, ADR-012.
3. [x] Rotate/inner-LOSO the val_subj choice instead of `pool[0]` for rigor
       (was a TODO in run_loso.py). — FULLY RESOLVED 2026-07-03, ADR-003
       (Option 1: rotate + median-select). Smoke-tested then launched full-scale.
4. [x] Launch the full 8-fold x 50-epoch inner-LOSO run, and promote its
       result to the new official baseline. — DONE 2026-07-02/03,
       RUN_20260702_202440 (3708s wall-clock, launched unattended via tmux),
       see §3. Supersedes RUN_20260702_172209 (retained on disk for
       comparison, not deleted).
5. [~] Implement nested split + grid search (kernel_size [3,5,7], dilations) with
       the RF guard (basic_tcn.receptive_field). — Task C from spec, SUPERSEDED
       by ADR-013 (Optuna TPE search, see item 10) rather than a plain grid --
       kernel_size/dilations restricted to 4 pre-vetted RF<=64 combos (K=7
       with any meaningfully-growing dilation set already exceeds RF=64 at
       window=64, so "grid over [3,5,7]" as originally scoped mostly collapses
       to K=3/K=5 anyway); TPE covers that same architecture axis plus 6 more
       hyperparameters (dropout, focal alpha/gamma, lr, weight_decay,
       batch_size) at once, with pruning a plain grid search can't do.
6. [ ] Add S04/S10 False-Alarm-Rate (FAR) suite (excluded from LOSO, ADR-002).
7. [x] Label-shift to prediction, horizon = 1 s. — ADR-006, implemented +
       VALIDATED at full scale as ADR-014. task="prediction" added
       additively to run_loso.main()/run_split() and
       FoGPreprocessor.create_windows() (DETECTION path unchanged, default,
       re-verified after every change to the shared code). 1-fold SMOKE done
       2026-07-03 (test=S01, n_epochs=10, RUN_20260703_160327), then FULL
       8-fold run DONE 2026-07-03 (n_epochs=50, RUN_20260703_173337,
       3664.1s wall-clock, launched detached via tmux session "predbase") —
       see §3 "PREDICTION BASELINE" and results/reports/01_baselines/baseline_prediction.txt.
       Episode-level recall/lead-time established as the PRIMARY metric for
       this task (window-level F1/PR-AUC are sparsity-limited, see §3). NOTE:
       docs/case_study_S08.md flagged that akinetic onsets (S08-type) might
       give even less pre-freeze lead-in signal than trembling onsets — the
       full run shows the OPPOSITE: S08 is NOT below chance under PREDICTION
       (unlike DETECTION), see §3.
8. [x] Add episode-level detection rate + latency (PR-AUC/ROC-AUC/F1 now in
       run_loso output). — ADR-007, implemented alongside ADR-014's
       episode_level_metrics() (run_loso.py): per-onset recall + mean lead
       time (seconds). VALIDATED at full 8-fold scale 2026-07-03: mean
       episode_recall=0.6981±0.1606 (pooled 0.7339), mean lead time=
       0.7154±0.0700s of the 1.00s horizon — now the PRIMARY metric for the
       PREDICTION task (see §3, results/reports/01_baselines/baseline_prediction.txt).
9. [ ] Wire external dataset (DeFOG / O'Day) as cross-dataset test. — ADR-008
10. [~] Optuna hyperparameter tuning to push the DETECTION baseline. — ADR-013
       (original ROC-AUC smoke) + ADR-018 (scaled-up PR-AUC overnight study,
       LAUNCHED). new src/tune_optuna.py, reuses run_loso.py's run_split()/
       inner_val_score() (not reimplemented). Search space UNCHANGED: 4
       RF<=64 (kernel_size, dilations) combos + dropout/focal alpha/focal
       gamma/lr/weight_decay/batch_size; TPE sampler + MedianPruner.
       SMOKE (ADR-013, ROC-AUC objective) done 2026-07-03 (2 trials x 1 fold
       x 5-epoch cap, results/reports/02_optuna_tuning/optuna_smoke.txt): RF guard rejects
       the known-bad combo, objective confirmed val-only, no leakage,
       trials persisted (sqlite + CSV). Found + fixed a real checkpoint-path
       collision bug along the way (shared hardcoded 'best_model.pth' across
       concurrent training runs -- see ADR-013 for detail); now every
       run_split() call uses a unique checkpoint path.
       [x] LAUNCHED 2026-07-04 (ADR-018): scaled-up OVERNIGHT study, two
       substantive changes from the smoke -- (1) objective = mean val
       PR-AUC (new inner_val_score_prauc(), NOT run_loso.py's ROC-AUC-based
       inner_val_score(), which stays unchanged for the production
       DETECTION pipeline/ADR-003); rationale: this project's real weakness
       is precision under imbalance, which ROC-AUC under-weights; (2)
       tuning_folds=[S01,S03,S08] (S08 deliberately swapped in for S02 --
       the akinetic outlier below-chance under DETECTION, ADR-012 -- so the
       search favors phenotype-generalizing params, not just easy
       subjects). Scale: 40 trials (up from smoke's 2), epoch_cap=25 (up
       from 5), MedianPruner n_startup_trials=8 (up from 5). PR-AUC SMOKE
       gate (2 trials x S01 x 5-epoch cap) run and verified interactively
       BEFORE launch: RF guard, val-PR-AUC objective, VAL-ONLY leakage
       guard (0 AssertionErrors across 14 models), trial persistence all
       CONFIRMED PASSED -- see results/reports/02_optuna_tuning/optuna_launch.txt sec 2.
       Wall-clock: ~8.03hr conservative upper bound (smoke's 5-epoch cap
       couldn't exercise early stopping at all, so real wall-clock should
       be meaningfully lower). Launched detached in tmux session "optuna"
       -> results/reports/06_run_logs/optuna_full.log, confirmed alive at launch time
       (config matches spec exactly in the log header). Study:
       results/optuna/study_full_prauc.db (sqlite, resumable); per-trial
       CSV: results/optuna/trials_log_full_prauc.csv. NOT done tonight
       (deferred as instructed): the final 50-epoch run with best params
       (run_finalize(), already implemented) and precision-aware threshold
       work. Full report: results/reports/02_optuna_tuning/optuna_launch.txt.
11. [ ] Investigate the tuned DETECTION candidate's (RUN_20260705_151423)
       F1/threshold-transfer inconsistency: 3 of 8 folds (S06, S07, S08)
       have LOW Test_F1 despite good-to-great Test_ROC_AUC under trial26's
       hyperparameters, while ROC-AUC/PR-AUC improved broadly. Candidate
       directions: re-sweep the val threshold-selection rule (Youden's J)
       under these hyperparameters specifically, or check whether
       dropout=0.19/focal_alpha=0.75/focal_gamma=2.02 shift the probability
       distribution's shape enough that a single frozen-threshold rule
       transfers less reliably than under the original baseline's
       hyperparameters. BUMPED UP (2026-07-06): still fully unresolved, and
       now the most concrete open correctness question on the books now
       that Jetson deployment infrastructure (item 13 below) is done. See §3
       "TUNED DETECTION baseline CANDIDATE" and
       results/reports/01_baselines/baseline_tuned.txt §3/§5.
12. [ ] Investigate PREDICTION's pre-FoG positive-window sparsity (~1-4% of
       windows, 5-15x sparser than DETECTION's ~19%) as the main limiter on
       window-level F1/PR-AUC, and S09's specifically low episode_recall
       (0.4074) despite near-best window-level PR-AUC/F1 -- candidate
       directions: shorter horizon / finer step_size to yield more positive
       windows per onset, oversampling or a recall-weighted loss for the
       pre-FoG class, or synthetic/augmented pre-FoG examples (time-warp,
       jitter) as a dataset-augmentation angle once the sparsity is confirmed
       to be the binding constraint rather than a signal-quality one. See §3,
       §5, results/reports/01_baselines/baseline_prediction.txt §3-4. Also
       still open: the "most cue triggers not tied to known onsets"
       precision flag from item 13's edge scaffolding work below -- same
       underlying window-level precision weakness.
13. [~] Deploy the PREDICTION model (RUN_20260703_173337) for closed-loop RAS
       cueing on Jetson Orin NX. — ADR-010 (edge target) + ADR-016 (new,
       deployment plan). Hardware-independent pipeline SCAFFOLDED 2026-07-03
       on WSL/RTX4060, no physical sensor: new edge/ package (model_utils.py
       shared config/loading + export_onnx.py + stream_infer.py + ras_cue.py
       + latency_harness.py), imports READ-ONLY from src/, src/ itself
       unmodified. Default deployment checkpoint = S01-fold of
       RUN_20260703_173337 (best_model_S01.pth), chosen specifically so the
       S01 replay is a genuine held-out sanity check, not a final full-data
       model (see ADR-016 / edge/model_utils.py).
       - [x] edge/stream_infer.py: ring-buffer streaming replay of
             data/dataset/S01R01.txt, causal by construction, reuses
             causal_median/causal_majority_vote from src/run_loso.py
             unchanged. 58 cue trigger events; 13/18 onsets preceded by a
             cue within the 1.0s horizon (sanity recall=0.7222, mean lead
             0.6899s) vs the offline reference (episode_recall=0.7826,
             mean_lead_time_s=0.6797 for full S01) -- same ballpark, expected
             deltas documented (single-file replay vs both S01 files,
             continuous-through-freeze streaming vs offline's excluded
             freeze windows). See results/reports/04_edge_deployment/edge_scaffold.txt sec 2.
       - [x] edge/ras_cue.py: RAS cueing stub, CueStrategy interface
             (FixedTempoCueStrategy implemented; AdaptivePhaseShiftCueStrategy
             a documented NotImplementedError stub for later). A real
             window-indexed-vs-sample-indexed timestamp bug was found and
             fixed during this session's own integration smoke test (not
             left for tomorrow) -- see results/reports/04_edge_deployment/edge_scaffold.txt sec 3.
       - [x] edge/export_onnx.py: UNBLOCKED 2026-07-03 -- installed
             onnx==1.22.0 + onnxruntime==1.27.0 (added to requirements.txt,
             not silent), ran the export: edge/artifacts/fog_tcn.onnx
             (opset=17, fixed input [1,3,64], sigmoid baked in), validated
             against 100 real S01 windows via onnxruntime CPU EP.
             **PyTorch vs onnxruntime max abs diff = 1.192e-07** (<< 1e-4
             threshold, no tolerance loosened). See
             results/reports/04_edge_deployment/edge_scaffold_2.txt sec A.
       - [x] edge/stream_infer.py: gained `--infer_step` (default 8, ADR-017),
             DECOUPLED from the training stride STEP_SIZE=32 (src/ untouched;
             64-sample causal window + model weights unchanged; only forward-
             pass cadence changes). Sweep on the S01R01.txt replay (18
             onsets), same causal post-proc reused unchanged from
             src/run_loso.py:
               infer_step=32: decision_interval=500ms, sanity_recall=0.7222
                 (13/18), mean_lead=0.6899s, cue_triggers=58
               infer_step=16: decision_interval=250ms, sanity_recall=0.8333
                 (15/18), mean_lead=0.8760s, cue_triggers=82
               infer_step=8:  decision_interval=125ms, sanity_recall=1.0000
                 (18/18), mean_lead=0.8490s, cue_triggers=226
             Smaller infer_step clearly improves recall/lead time on this
             replay, but the SAME k=5/w=7 causal post-proc now spans much
             less real time (combined worst-case span 5500ms->1375ms),
             which is most of why cue_triggers rises so sharply -- flagged
             as OPEN, not resolved (re-tune k/w or add a refractory period
             in ras_cue.py before picking a final infer_step for a live
             demo). See results/reports/04_edge_deployment/edge_scaffold_2.txt sec B/D.
       - [x] edge/latency_harness.py: now reports forward (MEASURED,
             ~1.2-1.7ms mean, empirically flat across infer_step -- confirms
             per-call compute cost doesn't change with cadence) and
             decision_interval (STRUCTURAL, 500/250/125ms per infer_step
             setting) as two SEPARATE, clearly labeled numbers -- the
             earlier "end_to_end" merged framing (edge_scaffold.txt part 1)
             is superseded, see ADR-017 for why merging them was misleading.
             Same harness reruns unchanged on Jetson tomorrow.
       - [x] RESOLVED 2026-07-04: span-constant post-proc + refractory
             debounce (ADR-017 extension). edge/stream_infer.py's k/w are
             now DERIVED per infer_step (derive_kw(), target
             median_span_ms=800/majority_span_ms=1000) so real-time
             smoothing span stays ~constant instead of collapsing;
             edge/ras_cue.py's RASCueEngine gained refractory_ms (default
             2000ms). Re-swept infer_step in {32,16,8}:
               infer_step=32: k=2,w=2, cue_triggers=122 (117 w/ refractory),
                 sanity_recall=1.0000 (18/18), mean_lead=0.7240s
               infer_step=16: k=3,w=4, cue_triggers=125 (112 w/ refractory),
                 sanity_recall=1.0000 (18/18), mean_lead=0.8351s
               infer_step=8:  k=6,w=8, cue_triggers=134 (123 w/ refractory),
                 sanity_recall=0.9444 (17/18), mean_lead=0.8667s
             KEY FINDING: trigger-count explosion CONFIRMED FIXED (1.10x
             spread vs the old 3.90x). Recall does NOT keep improving with
             smaller infer_step once span is constant (already at ceiling
             at infer_step=32) -- most of the ORIGINAL "smaller infer_step
             helps recall" trend was a smoothing-span dilution artifact of
             the old fixed k=5/w=7, not a genuine cadence effect. Lead time
             DOES show a real, modest cadence benefit (0.72s->0.84s->0.87s).
             SEPARATE OPEN FLAG (not resolved by this change): even with
             refractory, 112-134 triggers remain against only 18 true
             onsets -- most are NOT clustered around known onsets, a
             precision problem in the underlying probability stream
             (consistent with this task's known low window-level PR-AUC,
             0.093+/-0.058), not something cadence/span/refractory tuning
             alone fixes. See §4 item 12 (sparsity-driven follow-up).
             RECOMMENDATION (for approval, not auto-committed as a new
             default): infer_step=16, k=3, w=4, refractory_ms=2000 for
             tomorrow's Jetson demo -- matches infer_step=32's ceiling
             recall with meaningfully better lead time; infer_step=8 does
             not clearly outperform it. Full numbers:
             results/reports/04_edge_deployment/edge_scaffold_3.txt.
       Full reports: results/reports/04_edge_deployment/edge_scaffold.txt (part 1) +
       results/reports/04_edge_deployment/edge_scaffold_2.txt (part 2) +
       results/reports/04_edge_deployment/edge_scaffold_3.txt (part 3, this update).
       - Jetson deployment infrastructure DONE 2026-07-05 (approved
         infer_step=16/k=3/w=4/refractory_ms=2000 config): env verified
         (JetPack 6.2/TRT 10.3.0/onnxruntime 1.18.0 CPU/numpy 1.21.5, no
         torch -- ONNX/TensorRT-only deployment path), fog_tcn.onnx
         converted to both FP32 and FP16 TensorRT engines (FP32 adopted,
         ADR-021), numerical parity vs source ONNX confirmed (max abs diff
         3.51e-05 << 1e-4), and the on-device latency + sanity replay
         harness run (18/18 onsets caught, ~2.2x faster forward pass than
         WSL/RTX4060). See §11 for the full numbers. Remaining items are
         explicitly DEFERRED, not forgotten:
           (a) INT8 quantization study -- needs a calibration dataset, not
               attempted yet.
           (b) Real IMU wiring -- hardware TBD, no physical sensor yet.
           (c) RAS cueing hardware integration (actual audio output) --
               deferred, ras_cue.py's FixedTempoCueStrategy is software-only
               so far.
         The precision flag (most triggers not tied to known onsets) also
         remains open, see §4 item 12.

## 5. Known risks
- S08 below chance: CLOSED as hard-but-valid akinetic phenotype, not a data
  defect — see §3, docs/case_study_S08.md, ADR-012. Kept in the official
  baseline mean; no further action unless the write-up wants a deeper
  phenotype-level study.
- RESOLVED: the official baseline now uses inner-LOSO val_subj selection
  (RUN_20260702_202440, §3); the old fixed `val_subj = pool[0]` run
  (RUN_20260702_172209) is superseded and kept only for comparison.
- "+4% F1" headline is at risk until ADR-003/004 are resolved.
- Switching datasets mid-project is a timeline risk — current plan keeps Daphnet
  primary (ADR-008).
- RESOLVED (found 2026-07-03 during Optuna smoke): run_split()/trainer_tcn.py
  used to save every training checkpoint to a single hardcoded 'best_model.pth'
  in the repo CWD. Fine as long as only one training ran at a time with a
  fixed architecture, but unsafe the moment two things vary: (a) architecture
  changes between calls (Optuna), or (b) two training processes run
  concurrently against the same repo (this actually happened — a second
  session was mid-run on the PREDICTION task at the same time). Fixed: every
  run_split() call now gets its own unique checkpoint path
  (results/_tmp_checkpoints/). No longer a risk for future concurrent runs
  from either task.
- PREDICTION's pre-FoG positive-window sparsity (~1-4% of windows, vs
  DETECTION's ~19%) is the MAIN LIMITER on window-level F1/PR-AUC for that
  task — intrinsic to horizon=window_size=64 + step_size=32 (each onset
  yields at most ~2 positive windows by construction), not a labeling
  defect (ADR-014). Episode-level recall/lead-time route around it as the
  primary metric, but the sparsity itself is unaddressed — flagged as §4
  item 11 for a future pass (shorter horizon/finer stride, class-weighted
  loss, or dataset augmentation of pre-FoG examples).
- S09 has a low PREDICTION episode_recall (0.4074, the weakest of 8 folds)
  despite near-best window-level PR-AUC/F1 for that same fold — a specific
  missed-onset pattern (likely short/fast-onset episodes specifically, not
  yet confirmed) rather than a general signal failure. Flagged as §4 item 11
  for follow-up; not yet diagnosed the way S08 was (docs/case_study_S08.md).

## 6. Repo structure — FINALIZED (restructured 2026-06-14)
```
115Daphnet_FoG/
├── data/                  Daphnet raw SxxRyy files (untouched)
├── results/               RUN_<timestamp>/ outputs (loso_summary, curves, ...) (untouched)
│   └── reports/            human-facing summary .txt (baseline_summary, wsl_smoke, ...) — new 2026-07-02
├── src/
│   ├── basic_tcn.py       TCN architecture          <- need n_conv_per_block
│   ├── fog_preprocessing.py  windowing/labeling     <- need stride/overlap/label map
│   ├── trainer_tcn.py     focal loss / early stop / threshold?
│   ├── run_loso.py        LOSO + threshold + post-proc (was model/run_tcn_v2.py)  <- ADR-003/004 live here
│   └── plot_fog.py         plotting (was plot_fog.py at repo root)
├── scripts/
│   └── check_env.py       merged env check (was model/checkfile.py; check_version.py archived as duplicate)
├── _archive/
│   ├── run_tcn.py          older runner (was model/run_tcn.py)
│   └── check_version.py    superseded duplicate env-check script
├── edge/                   (empty) Jetson streaming-inference path, TBD
├── docs/
│   ├── literature/references.md  (was references.md at repo root)
│   └── deployment/         (empty) Jetson/TensorRT deployment notes, TBD
├── tests/                  (empty) split/RF/label-mapping/causality tests, TBD (R9)
├── .claude/skills/lit-review/SKILL.md  (was lit-review_SKILL.md at repo root)
├── EXPERIMENTS.md          run log (new, empty template)
├── README.md               (new, empty template)
├── requirements.txt        frozen from fog_env_64 (active venv)
├── CLAUDE.md              agent memory (this set)
├── fog_env/, fog_env_64/  Windows venvs (ignore, kept intact)
├── fog_env_wsl/            WSL-native venv, torch 2.5.1+cu121, GPU-verified (ignore) — new 2026-07-02
└── tree.txt
```

**WSL-native training env (2026-07-02):** `fog_env_wsl` (`python3 -m venv`, WSL Ubuntu)
added additively alongside `fog_env_64` (not a replacement) so CLI + training run in one
shell — see DECISIONS.md ADR (env reproducibility) for the exact install command.
`torch.cuda.is_available()` == True, device = RTX 4060. `~/.bashrc` now exports
`PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` to avoid the prior multi-shell encoding issues.
Smoke test (1 fold, test=S01, val=S02, n_epochs=2) ran on `cuda`, RF=57<=64, threshold
picked on val — see `results/reports/05_smoke_tests/wsl_smoke.txt`.

**results/ convention (2026-07-02):** per-run training outputs (loso_summary,
loso_detailed_predictions, curves/) live in `results/RUN_<timestamp>/`, written there
automatically by `run_loso.py` (path logic unchanged). Hand-written / uploaded summary
`.txt` files live in `results/reports/` instead of loose at `results/` top level.

**What moved (2026-06-14 restructure):**
- `model/` -> `src/` (basic_tcn.py, fog_preprocessing.py, trainer_tcn.py unchanged; flat
  intra-`src/` imports still resolve because Python adds the script's own dir to
  `sys.path[0]`).
- `model/run_tcn_v2.py` -> `src/run_loso.py`; `model/run_tcn.py` -> `_archive/run_tcn.py`.
- `plot_fog.py` (root) -> `src/plot_fog.py`.
- `model/checkfile.py` (richer package-table env check) -> `scripts/check_env.py`;
  `check_version.py` (root, duplicate/simpler) -> `_archive/check_version.py`.
- `references.md` -> `docs/literature/references.md`; `lit-review_SKILL.md` ->
  `.claude/skills/lit-review/SKILL.md`.
- Path fixes (R12 — landmine paths replaced with `ROOT_DIR = Path(__file__).resolve().parents[1]`):
  - `src/plot_fog.py`: hardcoded `C:/Project/115Daphnet_FoG/data/dataset/S01R01.txt`
    -> `ROOT_DIR / "data" / "dataset" / "S01R01.txt"`; output `fog_waveform_S01R01.png`
    now written to `ROOT_DIR` explicitly.
  - `src/run_loso.py`: hardcoded `C:/Project/115Daphnet_FoG/data/dataset` ->
    `ROOT_DIR / "data" / "dataset"`; bare relative `"results"` for `run_dir` ->
    `os.path.join(ROOT_DIR, "results", ...)`.
  - `_archive/run_tcn.py` (legacy, archived) and `trainer_tcn.py`'s
    relative `best_model.pth` (CWD-relative, repo-root-relative under the new
    "run from repo root" convention) were left as-is — out of scope / not landmines
    under the new convention.
- `model/__pycache__` and `model/checkfile.py` removed (build artifact + merged file);
  `model/` directory removed once empty. `data/` and `results/` untouched.

**Smoke test (2026-06-14):** `import src.basic_tcn / src.fog_preprocessing / src.trainer_tcn`
from repo root OK (namespace package, no `__init__.py` needed). Dry-run of the
`run_loso` pipeline (1 fold: train=S02R01, val=S01R01, 1 epoch, CPU) completed
end-to-end — PASS. Only warning: pre-existing `torch.nn.utils.weight_norm`
deprecation notice (unrelated to this move; tracked separately per CLAUDE.md
engineering standards).

## 7. Roadmap (≈10 weeks to end of August)
- W1–2: lock prediction task + fix methodology (ADR-003/004/005); freeze a model.
- W3–5: Jetson Orin NX export (ONNX→TensorRT), streaming inference, IMU integration.
- W6–7: RAS closed-loop cueing (basic fixed-tempo first); measure cue lead time.
- W8–9: ablations (64 vs 32 Hz, feature channels, cross-dataset), figures, stats.
- W9–10: write-up + slides + defense prep.

## 8. To finalize next
- [x] Folder restructure (model/ -> src/, scripts/, _archive/, docs/literature/,
      docs/deployment/, edge/, tests/, EXPERIMENTS.md, README.md, requirements.txt) —
      done 2026-06-14, see §6.
- [ ] SKILL.md — repeatable procedures (run LOSO, add feature channel, grid search,
      export TensorRT, lit-review via nrp-literature MCP).
- [ ] Populate tests/ per R9 (subject-overlap, RF<=window, label-mapping, causal
      post-proc checks).

## 9. Edge deployment platform (Jetson) — 2026-07-04
Platform consolidated ahead of on-device work (ONNX->TensorRT conversion,
real IMU wiring, on-device latency per ADR-016/017). See ADR-019 for why
this specific hardware/access setup was chosen.

**Hardware:** Yahboom Jetson Orin NX SUPER Developer Kit. Root storage
expanded from the stock 27GB eMMC/SD to a 476GB SPCC NVMe SSD (467GB
usable) -- needed headroom for CUDA/TensorRT/JetPack toolchain + model
artifacts + logs. Power: 19V DC barrel jack is MANDATORY (the module does
not reliably boot/run on USB-C or underrated supplies). Video out: DP
(DisplayPort).

**OS / JetPack:** L4T R36.4.7 (JetPack 6.2), Ubuntu 22.04, CUDA 12.6,
Python 3.10.12. JetPack 6.x's TensorRT (10.x) supports ONNX opset 17 --
matches edge/export_onnx.py's exported opset exactly (ADR-016), so
edge/artifacts/fog_tcn.onnx should convert without an opset-downgrade step.

**Access:** SSH over Tailscale (zero-config, no campus VPN dependency) --
Jetson tailnet address 100.97.43.118, tailnet candy070405@. Sudo user on
the device: "stu". Wired ethernet fallback: 140.116.132.43 (NCKU campus
network, only reachable on-campus). Tailscale is the primary/default path
since it works identically regardless of which network either end is on.

**Workspace:** ~/fog_edge/{artifacts,scripts,logs,models} on the Jetson --
a clean, separate workspace from the device's previous user's ~/jps/
directory, which is explicitly NOT to be touched (not ours, unrelated
prior work).

**Location & continuity:** the Jetson physically stays at NCKU; all
on-device work continues remotely via Tailscale SSH from Kaohsiung -- the
setup does not require physical presence at the device for day-to-day
development, only for the initial hardware/storage/power setup already
done and for anything requiring physical access (re-flashing, hardware
faults, peripheral changes).

**Known issue (no functional impact confirmed):** Tailscale logs an
iptables-legacy connmark warning on this device. A system restart is
pending to clear it; SSH and basic connectivity have been confirmed
working despite the warning, so this is not currently blocking any work --
flagged here so it isn't mistaken for a new problem if it resurfaces after
the pending restart.

## 10. reports/ layout (topic-organized as of 2026-07-05)
`results/reports/` was reorganized from a flat ~26-file directory into
topic subfolders. Full reorg record (mapping, diffs, before/after
reference counts, one data-loss disclosure): results/reports/99_misc/reorg_report.txt.

- `01_baselines/` -- DETECTION/PREDICTION baseline summary reports
  (baseline_*.txt, full_loso_report.txt).
- `02_optuna_tuning/` -- Optuna study reports (optuna_launch.txt,
  optuna_summary.txt, optuna_smoke.txt).
- `03_case_studies/` -- per-subject diagnostic deep-dives (diag_S08_report.txt
  + its diag_S08/ figure folder).
- `04_edge_deployment/` -- edge/Jetson deployment reports (edge_scaffold*.txt,
  jetson_*.txt).
- `05_smoke_tests/` -- 1-fold/quick smoke-test reports (wsl_smoke.txt,
  innerloso_smoke.txt, prediction_smoke.txt).
- `06_run_logs/` -- raw console logs from detached tmux training runs
  (*.log) -- git-ignored (results/reports/06_run_logs/*.log), kept locally
  only; the human-facing summary .txt for each run lives in the matching
  topic subfolder above instead.
- `99_misc/` -- ungrouped/meta reports (e.g. this reorg's own record).

**Rule for new reports:** put each new report in the subfolder matching its
topic above; raw run logs (`*.log`) go in `06_run_logs/` (git-ignored) --
NOT at `results/reports/` root and NOT alongside the `.txt` summary they
back. Launcher scripts (`scripts/launch_*.sh`) already write their `*.log`
redirect targets into `06_run_logs/` directly.

## 11. Jetson deployment measured performance (2026-07-05)
Deployment target: Jetson Orin NX SUPER (JetPack 6.2). Full per-task
reports: `results/reports/04_edge_deployment/jetson_env.txt` (env verify),
`jetson_scp.txt` (artifact transfer), `jetson_trt_build.txt` (engine
build), `jetson_verify_engine.txt` (numerical parity), `jetson_latency.txt`
(on-device latency + sanity replay).

**Environment:** JetPack 6.2, TensorRT 10.3.0, ONNX Runtime 1.18.0 (CPU
execution provider), NumPy 1.21.5. No PyTorch on this Jetson by design --
the deployment path is ONNX/TensorRT-only (saves the Jetson-specific torch
wheel install entirely; see ADR-021 for why this is sufficient).

**Deployment target: `fog_tcn_fp32.engine`.** Both FP32 and FP16 TensorRT
engines were built from `fog_tcn.onnx`; FP16 was NOT adopted -- it shows no
speedup on this model size (Task 3 / `jetson_trt_build.txt`: FP16 GPU
Compute time is statistically indistinguishable from FP32's, this model is
too small/launch-bound for FP16 to help). See ADR-021.

**Latency** (from `jetson_latency.txt`, primary deployment config
infer_step=16/k=3/w=4):

| metric | value |
|---|---|
| TRT forward, median | 0.6292 ms (P95 0.6415 ms) |
| Post-proc, median | 1.1921 ms (pandas call overhead dominates the tiny forward pass) |
| End-to-end compute, median | 1.8233 ms |
| Decision interval (structural) | 250 ms (= infer_step/fs_hz, NOT measured -- see jetson_latency.txt) |
| Compute as % of decision interval | ~0.73% |

**Sanity replay:** 18/18 known onsets in the S01R01.txt replay preceded by
>=1 cue within the 1.0s horizon (sanity_recall=1.0000); mean lead time
0.8906s, within 6.6% of the WSL/edge_scaffold_3.txt reference (0.8351s for
the same k/w/infer_step config) -- attributed to a scaler-fit difference
between the two harnesses (documented in `jetson_latency.txt`), not a
pipeline defect.

**Numerical parity vs source ONNX (Task 4):** TRT FP32 max abs diff =
3.51e-05, comfortably under the 1e-4 threshold (no tolerance loosened).

**Speedup vs WSL/RTX4060:** ~2.2x faster forward pass than the WSL PyTorch-
GPU baseline (1.4006 ms median, `edge_scaffold_2.txt`), and ~1.9x faster
than the WSL CPU baseline (1.2037 ms median).

---

## Session Updates 2026-07-09 to 2026-08-09

### Literature Review MCP Infrastructure (2026-07-09)
- Registered two project-scoped MCP servers in .mcp.json:
  scholar (ScholarMCP: OpenAlex + Crossref + Semantic Scholar + Google Scholar)
  scipapers (@futurelab-studio/latest-science-mcp: arXiv + OpenAlex + EuropePMC + CORE)
- .env.mcp.example created; .env.mcp gitignored via existing .env.* glob
- .claude/skills/lit-review/SKILL.md re-pointed from nrp-literature to live
  server names (0 remaining nrp-literature references)
- Raw query report: results/reports/07_literature/lit_review_imu_edge_20260709.txt
- 14 new references appended to docs/literature/references.md in two batches:
  Batch 1 (MCP search): Abbasi&Rezaee2024, Delgado-Teran2025, Gregorcic2025,
    Koltermann2024, Kita2017, Al-Adhaileh2025b
  Batch 2 (WOS manual): Djuric-Jovicic2014, Hwang2025, Yang2024, Sigcha2024,
    Ghai2018, Nieuwboer2007, DelDin2016, Costa2026
- Key confirmed technical parameters (see docs/lit_reading_strategy.md):
  Yang 2024: RF=243 samples=3.8s, non-causal, offline annotation only
  Sigcha 2024: Daphnet AUROC 0.844 (single-dataset); cross-dataset 0.839;
    domain shift root cause = subject heterogeneity, NOT Fs or sensor placement
  Hwang 2025: subject-dependent 70/30 CV -- 96% sensitivity NOT comparable to LOSO
  O'Day 2022: hardware 128Hz -> model 64Hz (FIR downsampled, confirmed)

### IMU Hardware Bring-up (2026-07-13)
- LSM6DS3 breakout board soldered and wired to Jetson I2C bus 7, addr 0x6A
  WHO_AM_I = 0x69 (confirmed valid for this board variant)
- Stable 104Hz acquisition via smbus2; scipy.signal.resample_poly(UP=8, DOWN=13)
  achieves 64Hz output (achieved: 64.9-65.0Hz steady-state, within 1% tolerance)
- imu_collector.py (Phase A): 104Hz acquisition thread + FIR resample + CSV logging
- imu_phase_b.py (Phase B): full pipeline with TRT FP32 engine inference
  DEPENDENCY CHAIN RESOLVED:
    pycuda needed numpy<2 -- pinned via pip install "numpy<2" --user
    onnxruntime 1.19.2 retained (1.23.x has ARM CPU vendor detection crash on Orin)
    scipy upgraded to 1.15.3 (numpy 2 compatible)
  TRT API MIGRATION: execute_async_v2 -> execute_async_v3 (TRT 10.x on JetPack 6.2)
  Tensor binding via set_tensor_address() (v3 API pattern)
- CURRENT BLOCKER: prob stuck at 0.573-0.578 regardless of motion
  Root cause: IMU axis orientation mismatch vs Daphnet training distribution
  (scaler mean/scale extracted from S02-S09 training fold and SCP'd to Jetson)
  Fix pending: need to measure static IMU values and remap axes to match Daphnet

### BLE IMU Architecture Decision (2026-08-09)
- Hardware: Seeed XIAO nRF52840 Sense ordered (LSM6DS3TR-C onboard, 21x17.5mm)
- Rationale: I2C jumper wire causes motion artifacts during walking tests;
  BLE eliminates mechanical coupling between sensor and Jetson
- nRF52840 Arduino firmware written (fog_imu_ble.ino):
  104Hz acquisition -> batch-4 BLE Notify (26-byte packets, 0xFE sync header)
  WHO_AM_I accepts 0x6C (TR-C variant) or 0x69 (original LSM6DS3)
- Jetson-side bleak receiver stub ready (drop-in for smbus2 acquisition thread)
- Handoff: PhD student (lab senior) handling BLE firmware on nRF-based board;
  will adapt to XIAO nRF52840 Sense. Interim: continue with I2C wired setup.

### CII Window Exclusion Analysis (2026-08-09)
ADR FINDING (no code change needed):
- Proposed CII-style window exclusion (Hwang 2025) is ALREADY implemented in
  the current PREDICTION pipeline via exclude_mask = (y == 1) in
  make_prediction_labels(), applied unconditionally in create_windows().
- All windows whose input frames contain active FoG are already excluded from
  both training and validation. The PREDICTION baseline (RUN_20260703_173337)
  is mathematically equivalent to Hwang's CII framing.
- Write-up framing: cite Hwang 2025 CII and note our exclude_mask implements
  the same principle. No new experiment required.
- True Hwang-inspired experiment NOT yet run: step_size=16 (25% overlap, vs
  current step_size=32 = 50%) -- deferred to Future Work per scope freeze.

### Scope Freeze (2026-08-15)
[SCOPE FROZEN 2026-08-15]
Hard deadline: 2026-09-07 (showcase demo)
From 2026-08-15: no new models, no new ADRs, no new features.
All future ideas -> future_work.md only.
Only allowed: fix IMU axis alignment bug (current demo blocker).
Demo target: causal TCN + single-ankle IMU + Jetson TRT + RAS metronome cue.
RAS target level: bonus tier (pygame 2Hz metronome, 5s duration).

### step_size=16 PREDICTION Baseline (2026-08-13) — NEW OFFICIAL
Run: RUN_20260812_162956
Replaces: RUN_20260703_173337 (step=32) as official PREDICTION reference.
Both runs retained on disk for comparison.

Results (8-fold nested LOSO, step=16, Trial-26 hyperparameters):
  Episode_Recall         : 0.7834 +/- 0.1069  (+0.085 vs step=32)
  Episode_MeanLeadTime_s : 0.8272 +/- 0.0604  (+0.112 vs step=32)
  Test_ROC_AUC           : 0.8126 +/- 0.0490  (+0.072 vs step=32)
  Note: all 8/8 folds contributed a valid Test_ROC_AUC (no NaN/excluded
  folds); per-fold positive window counts (Episode_N) range 10-65 -- see
  baseline_prediction_step16.txt for the full per-subject table.

Scientific rationale: step=16 aligns training stride with infer_step=16
(ADR-017). Consistent with Hwang 2025 overlap sensitivity analysis.
Improvement is structural (train-deploy alignment), not just data quantity.

Showcase primary numbers updated — see results/reports/01_baselines/showcase_data.md
