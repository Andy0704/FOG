"""Export the PREDICTION-task BasicTCN checkpoint to ONNX for TensorRT
conversion on Jetson (tomorrow) -- ADR-016. TASK: PREDICTION (horizon=1.0s
pre-FoG, ADR-014), NOT detection.

Usage:
    python edge/export_onnx.py [--ckpt PATH] [--run_dir PATH] [--out PATH]
                                [--n_validate N] [--subject S01]

Default checkpoint / rationale: see edge/model_utils.py (S01-fold of
RUN_20260703_173337, chosen so the validation + stream_infer.py replay on
S01 is a genuine held-out check).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model_utils import (
    WINDOW_SIZE, N_CHANNELS, ARTIFACTS_DIR, DEFAULT_CKPT,
    DEFAULT_REPLAY_SUBJECT, load_model, load_replay_samples,
    build_deployment_scaler,
)

ONNX_OPSET = 17  # widely supported by TensorRT 8.5+ / JetPack 5.x-6.x


class SigmoidWrapper(nn.Module):
    """Wraps BasicTCN so the exported ONNX graph outputs a probability
    directly (sigmoid baked in) -- simplifies stream_infer.py/
    latency_harness.py, which would otherwise have to replicate the sigmoid
    on the onnxruntime output separately."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def export(ckpt_path=None, out_path=None, n_validate=100, subject=DEFAULT_REPLAY_SUBJECT):
    ckpt_path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    out_path = Path(out_path) if out_path else (ARTIFACTS_DIR / "fog_tcn.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[TASK] PREDICTION (horizon=1.0s pre-FoG, ADR-014) -- NOT detection")
    print(f"[EXPORT] checkpoint={ckpt_path}")
    model = load_model(ckpt_path)
    wrapped = SigmoidWrapper(model)
    wrapped.eval()

    dummy = torch.randn(1, N_CHANNELS, WINDOW_SIZE, dtype=torch.float32)
    try:
        torch.onnx.export(
            wrapped, dummy, str(out_path),
            input_names=["imu_window"], output_names=["fog_prob"],
            opset_version=ONNX_OPSET,
            dynamic_axes=None,  # fixed shape [1, n_channels, 64] -- Jetson/TensorRT prefers a static engine
        )
    except Exception as e:
        print(f"[EXPORT] STOP: torch.onnx.export failed -- {type(e).__name__}: {e}")
        print("[EXPORT] This environment (fog_env_wsl) is missing the 'onnx' package")
        print("[EXPORT] (required by torch.onnx.export itself, not just validation).")
        print("[EXPORT] Install with:")
        print("[EXPORT]   pip install onnx onnxruntime")
        print("[EXPORT] Not installed automatically -- rerun this script after installing.")
        return None

    print(f"[EXPORT] saved: {out_path} (opset={ONNX_OPSET}, fixed input shape "
          f"[1, {N_CHANNELS}, {WINDOW_SIZE}])")

    # --- Validation: PyTorch vs onnxruntime on N real (scaled) windows ---
    try:
        import onnxruntime as ort
    except ImportError:
        print("[EXPORT] STOP: onnxruntime is not installed in this environment.")
        print("[EXPORT] Install with:")
        print("[EXPORT]   pip install onnxruntime")
        print("[EXPORT] (CPU execution provider is sufficient for this parity check;")
        print("[EXPORT] onnxruntime-gpu is not required for validation.)")
        print("[EXPORT] Not installed automatically -- rerun this script after installing.")
        return None

    X_raw, y_raw, src_file = load_replay_samples(subject)
    scaler = build_deployment_scaler()
    X_scaled = scaler.transform(X_raw)
    n_windows_available = len(X_scaled) - WINDOW_SIZE + 1
    n = min(n_validate, n_windows_available)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    with torch.no_grad():
        for start in range(n):
            window = X_scaled[start:start + WINDOW_SIZE]          # (64, 3)
            x_np = window.T.astype(np.float32)[None, :, :]        # (1, 3, 64)
            x_torch = torch.from_numpy(x_np)
            torch_out = wrapped(x_torch).numpy()
            onnx_out = sess.run(None, {"imu_window": x_np})[0]
            diff = float(np.max(np.abs(torch_out - onnx_out)))
            max_diff = max(max_diff, diff)

    print(f"[EXPORT] validated {n} windows (subject={subject}) -- "
          f"PyTorch vs onnxruntime max abs diff = {max_diff:.3e}")
    ok = max_diff < 1e-4
    print(f"[EXPORT] PARITY CHECK {'PASSED' if ok else 'FAILED'} (threshold 1e-4)")
    assert ok, f"ONNX/PyTorch parity check failed: max_diff={max_diff} >= 1e-4"
    return max_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                         help="Path to .pth checkpoint; default = S01-fold of RUN_20260703_173337 (see model_utils.py)")
    parser.add_argument("--run_dir", default=None,
                         help="Override run dir; uses <run_dir>/models/best_model_S01.pth if --ckpt not also given")
    parser.add_argument("--out", default=None, help="Output .onnx path; default = edge/artifacts/fog_tcn.onnx")
    parser.add_argument("--n_validate", type=int, default=100)
    parser.add_argument("--subject", default=DEFAULT_REPLAY_SUBJECT)
    args = parser.parse_args()

    ckpt = args.ckpt
    if ckpt is None and args.run_dir is not None:
        ckpt = str(Path(args.run_dir) / "models" / "best_model_S01.pth")

    export(ckpt_path=ckpt, out_path=args.out, n_validate=args.n_validate, subject=args.subject)
