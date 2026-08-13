#!/bin/bash
# PREDICTION task with step_size=16 (25% overlap, matches infer_step=16)
# Hwang 2025 overlap sensitivity: overlap < 25% reduces overfitting risk
# Baseline comparison: step_size=32 -> results/reports/01_baselines/baseline_prediction.txt
#
# NOTE: main() in src/run_loso.py does not expose a step_size parameter
# (only run_split()/FoGPreprocessor do). Per explicit sign-off, step_size=16
# is threaded through via a runtime monkeypatch of run_loso.run_split, done
# entirely in this script -- zero writes to src/. Smoke-tested (2 folds,
# 2 epochs): window counts doubled exactly vs the step=32 baseline
# (S01 pos 46->92, S02 pos 45->90), 0 NaN/leakage assertions.
set -e
cd /mnt/c/Project/115Daphnet_FoG
source fog_env_wsl/bin/activate
python -u -c "
import sys; sys.path.insert(0, 'src')
import run_loso
_orig_run_split = run_loso.run_split
def _run_split_step16(*args, **kwargs):
    kwargs.setdefault('step_size', 16)
    return _orig_run_split(*args, **kwargs)
run_loso.run_split = _run_split_step16
from run_loso import main
main(
    task='prediction',
    horizon=64,
    n_epochs=50,
    subjects_to_run=None,
    kernel_size=5,
    dilations=[1,2,4],
    dropout=0.2,
    alpha=0.75,
    gamma=2.0,
    lr=0.000776,
    weight_decay=0.0001,
    batch_size=256,
    seed=42,
)
" 2>&1 | tee results/reports/06_run_logs/prediction_step16_$(date +%Y%m%d_%H%M%S).log
