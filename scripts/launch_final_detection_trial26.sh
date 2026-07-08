#!/usr/bin/env bash
# Launches the full 8-fold inner-LOSO DETECTION run using Optuna trial 26's
# CLEAN hyperparameters (ADR-018/ADR-020) -- the tuned-baseline candidate to
# compare against RUN_20260702_202440. n_epochs=50, seed=42, cuda.
#
# WHY TRIAL 26 NOT TRIAL 33: trial 33's nominal top val PR-AUC (0.6260) came
# from a NaN-contaminated run -- its S08-fold selected candidate diverged at
# epoch 16 and was checkpoint-rescued. Trial 26 is the best CLEAN trial (no
# divergence anywhere across all 21 candidate trainings) at val PR-AUC
# 0.6173, a gap of only 0.0087 -- within the top-5 spread (0.0113). See
# results/reports/02_optuna_tuning/optuna_summary.txt and DECISIONS.md ADR-018/ADR-020.
#
# Mirrors scripts/launch_full_inner_loso.sh exactly (same main(), no logic
# duplicated) -- only the hyperparameters differ.
# Meant to be run detached inside tmux so it survives an SSH/session close:
#   tmux new-session -d -s final_det "bash scripts/launch_final_detection_trial26.sh > results/reports/06_run_logs/final_det.log 2>&1"
set -e
cd "$(dirname "$0")/.."
source fog_env_wsl/bin/activate
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
python -u -c "
import sys
sys.path.insert(0, 'src')
from run_loso import main
main(n_epochs=50, subjects_to_run=None, seed=42, task='detection',
     kernel_size=5, dilations=[1, 2, 4],
     dropout=0.188296, alpha=0.747900, gamma=2.020007,
     lr=0.000780, weight_decay=0.000699, batch_size=256)
"
