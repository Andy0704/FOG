# Showcase Data — FoG Edge Deployment Project
Generated: 2026-08-13
Hard deadline: 2026-09-07 (showcase demo)

## Primary Numbers (use these in poster, slides, and write-up)

### Detection Task (headline model quality)
- ROC-AUC        : 0.866 ± 0.073  (8-fold nested LOSO, Trial-26 Optuna)
- Run             : RUN_20260705_151423
- Report          : results/reports/01_baselines/baseline_tuned.txt

### Prediction Task (clinical relevance)  [step=16, NEW OFFICIAL BASELINE]
- Episode Recall  : 0.783 ± 0.107  (8-fold LOSO; step_size=16)
- Mean Lead Time  : 0.827 ± 0.060 s
- ROC-AUC         : 0.813 ± 0.049  (8-fold nested LOSO; all 8 folds valid)
- Run             : RUN_20260812_162956
- Report          : results/reports/01_baselines/baseline_prediction_step16.txt
- NOTE: step=16 aligns training stride with deployment infer_step=16 (ADR-017).
  Previous step=32 baseline (RUN_20260703_173337) retained for comparison.

### Edge Deployment
- Jetson end-to-end latency : 1.82 ms  (median, TRT FP32, ADR-021/022)
- Decision interval         : 250 ms   (infer_step=16 @ 64Hz)
- Compute utilization       : <1% of decision interval

## Comparison Table (for poster Section 4)
| Method              | IMU | Protocol    | Latency    | Causal | Lead time |
|---------------------|-----|-------------|------------|--------|-----------|
| **Ours**            | 1   | **LOSO**    | **1.82ms** | **Yes**| **0.83s** |
| Yang 2024 (JNER)    | 5   | LOSO        | needs 1.89s future | No | offline |
| Hwang 2025 (TNSRE)  | 3   | Subject-dep†| unreported | unknwn | ~2s pre-FoG |
| O'Day 2022 (JNER)   | 1   | LOSO        | unreported | unknwn | n/a |
| Koltermann 2024     | -   | Subject-indep| 378.5ms  | Yes    | n/a |

† Hwang 2025 sensitivity=96% uses subject-dependent 70/30 split —
  NOT comparable to LOSO numbers.

## Demo Script (3-minute version)
1. [0:00–0:45] Clinical motivation: PD → FoG → fall → RAS works but needs trigger
2. [0:45–1:30] Technical: causal TCN + single ankle + LOSO + 1.82ms Jetson
3. [1:30–3:00] Live demo: walk → prob curve stable → simulate freeze → prob rises
               → threshold crossed → 🔔 metronome → "this is the cue to keep walking"

## Key Differentiators (3 bullets for CV/推薦信)
1. Causal inference: no future data required (vs Yang 2024 non-causal 3.8s RF)
2. Strict LOSO validation: subject-independent (vs Hwang 2025 subject-dependent)
3. Edge-deployed: 1.82ms on Jetson Orin NX, 207x faster than Koltermann 2024

## Step=16 Improvement Summary
| Metric          | step=32 (old) | step=16 (new) | delta   |
|-----------------|---------------|---------------|---------|
| Episode_Recall  | 0.698 ± 0.161 | 0.783 ± 0.107 | +0.085  |
| Lead_Time_s     | 0.715 ± 0.070 | 0.827 ± 0.060 | +0.112s |
| Test_ROC_AUC    | 0.740 ± 0.086 | 0.813 ± 0.049 | +0.072  |
Scientific explanation: training stride (step=16) now matches deployment
infer_step=16 (ADR-017). Consistent with Hwang 2025 overlap sensitivity finding.
