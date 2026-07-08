"""WSL-baseline latency harness for the streaming PREDICTION pipeline --
ADR-016 / ADR-017. TASK: PREDICTION (horizon=1.0s pre-FoG, ADR-014), NOT
detection.

Measures, per forward-pass step:
  - forward_ms: real, MEASURED model-forward compute time (time.perf_counter
    around model(x); torch.cuda.synchronize() first if on GPU, so async
    kernel launches don't skew the number). Independent of infer_step -- the
    compute cost of a single forward pass does not change with how often it
    fires; this harness measures it once per infer_step setting anyway, so
    the sweep table below can show empirically that it IS stable.
  - decision_interval_ms = infer_step/fs_hz*1000: a fixed STRUCTURAL
    constant, NOT a measured time -- the real wall-clock gap between
    consecutive forward-pass opportunities in a live stream (a fresh window
    is only ready once `infer_step` new samples have arrived at fs_hz=64Hz).
    A pre-recorded in-memory replay iterates at CPU/array speed, not sensor
    speed, so this cannot be honestly *measured* via a timer tonight --
    reporting a fabricated "measured" number for something not actually
    being timed would be worse than reporting the true structural constant.
    On Jetson with a REAL IMU streaming at fs_hz, this becomes a genuinely
    measurable wait (same harness, real hardware, tomorrow).

FORWARD COMPUTE AND DECISION INTERVAL ARE REPORTED SEPARATELY AND ARE NOT
MERGED INTO A SINGLE "end_to_end" NUMBER -- that would hide which part is
actual GPU/CPU compute (µs-ms, hardware-dependent, what Jetson/TensorRT can
actually improve) versus which part is a structural consequence of the
chosen infer_step (ms, a design/config choice, not a hardware limit).
ADR-017 covers why infer_step is now decoupled from the training stride.

*** THESE ARE WSL/RTX4060 BASELINE NUMBERS, NOT JETSON FIGURES. *** The
exact same harness runs unchanged on Jetson tomorrow for the real numbers.

Usage:
    python edge/latency_harness.py [--subject S01] [--device cpu|cuda] [--n_warmup 20]
                                    [--infer_steps 32,16,8]
"""
import argparse
import time
from collections import deque

import numpy as np
import torch

from model_utils import (
    WINDOW_SIZE, STEP_SIZE, FS_HZ, DEFAULT_REPLAY_SUBJECT,
    load_model, load_replay_samples, build_deployment_scaler,
)


def _stats(arr):
    a = np.array(arr, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}


def measure_latency(subject=DEFAULT_REPLAY_SUBJECT, device="cpu", n_warmup=20,
                     fs_hz=FS_HZ, infer_step=8):
    """infer_step: samples between forward passes (deployment decision
    cadence, ADR-017) -- independent of the training stride STEP_SIZE=32.
    Returns forward_ms (measured) and decision_interval_ms (structural
    constant) as SEPARATE fields -- never pre-summed into one number."""
    model = load_model().to(device)
    scaler = build_deployment_scaler()
    X_raw, y_raw, src_file = load_replay_samples(subject)
    X_scaled = scaler.transform(X_raw)

    ring = deque(maxlen=WINDOW_SIZE)
    step_counter = 0
    forward_times_ms = []

    decision_interval_ms = (infer_step / fs_hz) * 1000.0  # structural constant, see module docstring

    with torch.no_grad():
        for i, sample in enumerate(X_scaled):
            ring.append(sample)
            step_counter += 1
            if len(ring) < WINDOW_SIZE or step_counter < infer_step:
                continue
            step_counter = 0

            window = np.stack(ring, axis=0)
            x = torch.tensor(window.T, dtype=torch.float32, device=device).unsqueeze(0)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logit = model(x)
            _ = torch.sigmoid(logit).item()  # .item() forces sync on CPU; explicit sync below covers CUDA
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            forward_times_ms.append((t1 - t0) * 1000.0)

    forward_times_ms = forward_times_ms[n_warmup:]  # drop cold-start/cudnn-autotune iterations
    fwd_stats = _stats(forward_times_ms)

    return {
        "forward_ms": fwd_stats,               # MEASURED
        "decision_interval_ms": decision_interval_ms,  # STRUCTURAL, not measured
        "infer_step": infer_step,
        "n_steps": len(forward_times_ms),
        "n_warmup": n_warmup,
        "device": device,
        "subject": subject,
        "src_file": src_file,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=DEFAULT_REPLAY_SUBJECT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_warmup", type=int, default=20)
    parser.add_argument("--infer_steps", default="32,16,8",
                         help="Comma-separated infer_step values to sweep (ADR-017); "
                              "default sweeps the same 3 settings as edge/stream_infer.py.")
    args = parser.parse_args()
    infer_steps = [int(s) for s in args.infer_steps.split(",")]

    print("[TASK] PREDICTION (horizon=1.0s pre-FoG, ADR-014) -- NOT detection")
    print(f"[LATENCY] *** WSL-BASELINE, NOT JETSON *** device={args.device} subject={args.subject}")
    print(f"[LATENCY] training STEP_SIZE={STEP_SIZE} (unrelated to infer_step below, ADR-017)")

    for infer_step in infer_steps:
        result = measure_latency(subject=args.subject, device=args.device,
                                  n_warmup=args.n_warmup, infer_step=infer_step)
        fwd = result["forward_ms"]
        print(f"\n[LATENCY] infer_step={infer_step}")
        print(f"[LATENCY][forward, MEASURED]              "
              f"mean={fwd['mean']:.4f}ms  median={fwd['median']:.4f}ms  p95={fwd['p95']:.4f}ms")
        print(f"[LATENCY][decision_interval, STRUCTURAL]  "
              f"{result['decision_interval_ms']:.4f}ms (= infer_step/fs_hz*1000, NOT measured -- "
              f"see module docstring)")
        print(f"[LATENCY] n_steps={result['n_steps']} (n_warmup={result['n_warmup']} dropped) "
              f"device={result['device']}")

    print("\n[LATENCY] *** WSL-BASELINE ONLY. Jetson figures pending tomorrow's hardware run "
          "(same harness, unchanged) -- see PROJECT_STATUS.md sec 4 / DECISIONS.md ADR-016/017. ***")
    print("[LATENCY] forward (compute) and decision_interval (structural) are reported SEPARATELY "
          "on purpose -- they are not the same kind of number and summing them would hide which "
          "part Jetson/TensorRT can actually improve (forward) vs which part is a config choice "
          "(decision_interval, set by infer_step).")
