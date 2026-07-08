#!/usr/bin/env bash
# Launches the full 8-fold inner-LOSO PREDICTION run (task="prediction",
# horizon=64 samples=1.0s, n_epochs=50, seed=42, cuda). Mirrors
# scripts/launch_full_inner_loso.sh (the DETECTION baseline launcher) --
# only the task/horizon kwargs differ. See ADR-014.
# Meant to be run detached inside tmux so it survives an SSH/session close:
#   tmux new-session -d -s predbase "bash scripts/launch_full_inner_loso_prediction.sh > results/reports/06_run_logs/prediction_full.log 2>&1"
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
main(n_epochs=50, subjects_to_run=None, seed=42, task='prediction', horizon=64)
"
