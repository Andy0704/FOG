"""Shared helpers for the edge/ deployment pipeline (ADR-016).

Imports READ-ONLY from src/ -- never modifies training logic, never touches
data/ or ROOT_DIR/logging. This module exists so export_onnx.py,
stream_infer.py, ras_cue.py and latency_harness.py don't each reimplement
"which checkpoint / which scaler / which architecture config" separately.

TASK: PREDICTION (label shifted +H=64 samples=1.0s before FoG onset, ADR-014)
-- NOT detection. All deployment artifacts here are for the PREDICTION model.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402
from basic_tcn import BasicTCN  # noqa: E402
from fog_preprocessing import FoGPreprocessor  # noqa: E402
from run_loso import (  # noqa: E402
    discover_subject_files, build_xy, load_daphnet_file,
    causal_median, causal_majority_vote, episode_level_metrics,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# --- Architecture / task config -- MUST match RUN_20260703_173337 exactly ---
WINDOW_SIZE = 64
STEP_SIZE = 32
KERNEL_SIZE = 5
DILATIONS = [1, 2, 4]
N_CHANNELS = 3
HORIZON = 64          # 1.0s @ 64Hz, PREDICTION task (ADR-014)
FS_HZ = 64.0

# --- Deployment checkpoint selection (see ADR-016 for the full rationale) ---
# RUN_20260703_173337 (the official PREDICTION baseline, ADR-014) trained one
# checkpoint PER OUTER FOLD -- each excludes its own test_subj, so none is a
# "deploy to any new patient" model; that would require a fresh full-data
# retrain (deliberately NOT done tonight -- out of scope, see ADR-016).
#
# Default = the S01-fold checkpoint (best_model_S01.pth, trained on
# train_subjs=[S03,S05,S06,S07,S08,S09], val_subj=S02, test_subj=S01 held
# out). Chosen -- over e.g. the S03 fold's higher raw ROC-AUC (0.8982) --
# specifically so that replaying S01 in stream_infer.py/latency_harness.py
# is a genuine HELD-OUT sanity check against the offline
# episode_recall=0.7826 for S01 (RUN_20260703_173337 CSV), not an optimistic
# non-held-out replay through a model that already saw S01 during training.
DEFAULT_RUN_DIR = ROOT_DIR / "results" / "RUN_20260703_173337"
DEFAULT_CKPT = DEFAULT_RUN_DIR / "models" / "best_model_S01.pth"
DEFAULT_REPLAY_SUBJECT = "S01"
DEFAULT_TRAIN_SUBJS = ["S03", "S05", "S06", "S07", "S08", "S09"]  # S01-fold's train_subjs
DEFAULT_BEST_THRESHOLD = 0.1763526350259781  # S01-fold frozen val threshold, loso_summary_20260703_173337.csv
OFFLINE_EPISODE_RECALL_S01 = 0.782608695652174    # same CSV row, for the stream_infer.py sanity comparison
OFFLINE_MEAN_LEAD_TIME_S01 = 0.6796875


def load_model(ckpt_path=None):
    """Load a BasicTCN checkpoint with the PREDICTION-model architecture.
    Read-only: only reads a .pth file, never writes to results/ or src/."""
    ckpt_path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    model = BasicTCN(n_channels=N_CHANNELS, n_classes=1, kernel_size=KERNEL_SIZE,
                      dilations=DILATIONS, window_size=WINDOW_SIZE)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def build_deployment_scaler(train_subjs=None):
    """Reconstruct the EXACT StandardScaler used to train DEFAULT_CKPT, by
    refitting on the same train_subjs run_loso.py used for that fold
    (deterministic given the same raw data -- this reproduces what training
    already did, it does NOT fit on new/held-out data; run_loso.py never
    persisted the fitted scaler object itself, only the model state_dict)."""
    train_subjs = train_subjs or DEFAULT_TRAIN_SUBJS
    subject_files = discover_subject_files(ROOT_DIR)
    X_train_raw, _ = build_xy(train_subjs, subject_files)
    preprocessor = FoGPreprocessor(window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    preprocessor.scale_train(X_train_raw)
    return preprocessor.scaler


def load_replay_samples(subject=DEFAULT_REPLAY_SUBJECT, root_dir=ROOT_DIR):
    """Load raw (0-filtered, binary-labeled) X, y for a single Daphnet
    subject's FIRST file, for replay simulation. Read-only reuse of
    src/run_loso.py's discover_subject_files/load_daphnet_file -- no
    src/ modification, no data/ modification."""
    subject_files = discover_subject_files(root_dir)
    f = subject_files[subject][0]
    X, y = load_daphnet_file(f)
    return X, y, f
