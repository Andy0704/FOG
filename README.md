# 115Daphnet_FoG

Real-time **Freezing of Gait (FoG) prediction** for Parkinson's patients from a
single wearable IMU, built as the sensing/decision layer of a closed-loop
neuro-rehabilitation system (RAS cueing). Core model is a causal, dilated
**Temporal Convolutional Network (TCN)** — chosen over LSTM/RNN variants for
low, deterministic inference latency on embedded hardware. Primary dataset is
**Daphnet FoG** (64 Hz, 10 subjects, 8 with FoG episodes). Target deployment
platform is a **Jetson Orin NX 8GB** with a single ankle-worn IMU.

The task is framed two ways, and the code/logs always say which:
- **DETECTION** — label at time *t* (is FoG happening now).
- **PREDICTION** — label shifted +1s into the future (will FoG start soon),
  the framing actually useful for a cueing device that must act *before*
  the freeze.

## Current status (see `PROJECT_STATUS.md` for the live, authoritative state)

Leak-free protocol: subject-level LOSO (leave-one-subject-out) splits, nested
inner-LOSO validation-subject selection, decision threshold frozen on the
validation subject and only then applied to test, causal post-processing
(median filter + rolling majority vote) for every number reported as
real-time.

| Track | Run | Key metrics (8-fold LOSO, mean ± std) |
|---|---|---|
| DETECTION baseline | `RUN_20260702_202440` | F1 0.366±0.144, PR-AUC 0.421±0.108, ROC-AUC 0.774±0.126 |
| DETECTION, tuned (candidate) | `RUN_20260705_151423` (Optuna trial 26) | ROC-AUC 0.866±0.073, PR-AUC 0.553±0.157, F1 0.387±0.220 — not yet promoted, threshold-transfer issue flagged |
| **PREDICTION baseline (official)** | `RUN_20260812_162956` (step=16, infer-aligned) | Episode recall 0.783±0.107, mean lead time 0.827±0.060s, ROC-AUC 0.813±0.049 |
| Jetson Orin NX, TensorRT FP32 | on-device replay | forward 0.63ms / end-to-end 1.82ms median, 18/18 onsets caught |

Episode-level recall and lead time are the primary metrics for PREDICTION
(window-level F1/PR-AUC are sparsity-limited artifacts of the ~1–4% pre-FoG
positive rate — see `EXPERIMENTS.md`). Full run-by-run numbers, ADRs behind
every methodological choice, and open follow-ups live in `PROJECT_STATUS.md`,
`DECISIONS.md`, and `EXPERIMENTS.md`.

## Repo structure

```
src/                  Core pipeline: preprocessing, TCN model, LOSO trainer, Optuna tuning
  fog_preprocessing.py   Windowing, labeling (detection/prediction), scaling, exclusion masks
  basic_tcn.py            Causal dilated-conv TCN model
  trainer_tcn.py           Training loop, loss (focal), metrics
  run_loso.py              Leave-one-subject-out orchestration (the entry point for a run)
  tune_optuna.py           Hyperparameter search (val-only objective)

edge/                 Hardware-independent deployment path (runs without a Jetson too)
  export_onnx.py           PyTorch -> ONNX export + parity check
  stream_infer.py          Streaming inference loop (task-agnostic: detection or prediction)
  ras_cue.py               RAS metronome cueing logic + refractory debounce
  latency_harness.py      Forward-pass / end-to-end latency measurement
  artifacts/               Exported ONNX/TensorRT artifacts (large binaries kept out of git)

scripts/              One-off launch scripts for specific runs (documents exact invocation/config)

results/
  RUN_<timestamp>/        One directory per training run: loso_summary*.csv, curves/, models/
                          (model weights + per-window prediction CSVs are gitignored — see .gitignore)
  reports/                 Curated, human-readable write-ups of specific runs, organized by topic:
    01_baselines/, 02_optuna_tuning/, 03_case_studies/, 04_edge_deployment/,
    05_smoke_tests/, 06_run_logs/, 07_literature/

docs/
  case_study_S08.md         Deep dive on the one akinetic-phenotype subject
  literature/references.md  Curated literature review (FoG / IMU / TCN / edge deployment)
  deployment/                Jetson/edge deployment notes

data/                  Not tracked in git (licensing) — see data/README for how to obtain Daphnet

CLAUDE.md             Project rules & non-negotiable scientific constraints (read this first)
PROJECT_STATUS.md     Live state: what's done, what's in flight, what's next
DECISIONS.md          ADRs — the "why" behind every methodological choice
EXPERIMENTS.md        One row per experiment: run_id, config, metrics, takeaway
```

## Reproducing a baseline

```bash
source fog_env_wsl/bin/activate   # or your own env — see requirements.txt
python -u -c "
import sys; sys.path.insert(0, 'src')
from run_loso import main
main(task='detection', n_epochs=50, subjects_to_run=None, seed=42)
"
```

See `scripts/` for the exact invocation (including hyperparameters) behind
each run_id in `EXPERIMENTS.md`.
