"""Ring-buffer streaming inference simulation for the PREDICTION model
(ADR-014 / ADR-016 / ADR-017). TASK: PREDICTION (horizon=1.0s pre-FoG), NOT
detection.

Consumes IMU samples ONE AT A TIME to simulate a live sensor -- tonight that
means replaying a Daphnet subject's raw file sample-by-sample (no physical
sensor, WSL only). Maintains a 64-sample causal window (unchanged -- this is
the model's fixed input shape) and applies the SAME causal post-proc
functions used in training (causal_median k=5, causal_majority_vote w=7,
imported unchanged from src/run_loso.py -- not reinvented here).

DECOUPLED DECISION CADENCE (ADR-017): a forward pass fires every
`infer_step` samples, a parameter INDEPENDENT of the training stride
STEP_SIZE=32 (STEP_SIZE only ever controlled how *training windows* were
carved out of a fixed dataset -- it has no reason to also gate how often a
DEPLOYED, already-trained model is allowed to look at a fresh window; the
model weights, the 64-sample window, and the training data/labels are all
completely unchanged by this parameter). Smaller infer_step -> shorter
real-time decision interval (infer_step/64*1000 ms) at the cost of more
forward passes per second (still ~1-2ms each on WSL, see
edge/latency_harness.py).

SPAN-CONSTANT POST-PROC (ADR-017 resolution): causal_median(k)/
causal_majority_vote(w) count DECISION steps, not raw samples, so a FIXED
k/w shrinks in real-time span as infer_step shrinks (this is what produced
the cue-trigger explosion documented in results/reports/04_edge_deployment/edge_scaffold_2.txt).
k and w are now DERIVED per infer_step from two target real-time spans
(median_span_ms, majority_span_ms) so the causal post-proc's real-time
reach stays approximately CONSTANT across infer_step settings instead of
collapsing:
    k = max(1, round(median_span_ms   / decision_interval_ms))
    w = max(1, round(majority_span_ms / decision_interval_ms))
Pass explicit k/w to override this derivation (e.g. to reproduce the old
fixed-k=5/w=7 behavior for comparison against edge_scaffold_2.txt).

CAVEAT (read before trusting the sanity numbers): the default checkpoint
(S01-fold of RUN_20260703_173337) was trained EXCLUDING S01, so replaying
S01 here is a genuine held-out check -- see edge/model_utils.py for why that
fold was picked specifically for this reason. If you swap in a different
checkpoint/subject combination where the subject WAS in that checkpoint's
training pool, the sanity numbers below are no longer a held-out comparison.

Usage:
    python edge/stream_infer.py [--subject S01] [--threshold T] [--infer_step 8]
                                 [--median_span_ms 800] [--majority_span_ms 1000]
                                 [--refractory_ms 2000] [--k K] [--w W]
"""
import argparse
from collections import deque

import numpy as np
import torch

from model_utils import (
    WINDOW_SIZE, STEP_SIZE, HORIZON, FS_HZ, DEFAULT_BEST_THRESHOLD,
    DEFAULT_REPLAY_SUBJECT, OFFLINE_EPISODE_RECALL_S01, OFFLINE_MEAN_LEAD_TIME_S01,
    load_model, load_replay_samples, build_deployment_scaler,
    causal_median, causal_majority_vote, episode_level_metrics,
)
from fog_preprocessing import FoGPreprocessor
from ras_cue import RASCueEngine


def derive_kw(infer_step, median_span_ms=800.0, majority_span_ms=1000.0, fs_hz=FS_HZ):
    """Derive causal_median's k and causal_majority_vote's w so their
    real-time span stays close to the target spans, for the GIVEN
    infer_step (ADR-017). Returns (k, w, decision_interval_ms,
    actual_median_span_ms, actual_majority_span_ms)."""
    decision_interval_ms = (infer_step / fs_hz) * 1000.0
    k = max(1, round(median_span_ms / decision_interval_ms))
    w = max(1, round(majority_span_ms / decision_interval_ms))
    actual_median_span_ms = k * decision_interval_ms
    actual_majority_span_ms = w * decision_interval_ms
    return k, w, decision_interval_ms, actual_median_span_ms, actual_majority_span_ms


def replay_stream(subject=DEFAULT_REPLAY_SUBJECT, threshold=DEFAULT_BEST_THRESHOLD,
                   k=None, w=None, infer_step=8, median_span_ms=800.0, majority_span_ms=1000.0,
                   refractory_ms=2000.0, verbose=True):
    """infer_step: samples between forward passes (DEPLOYMENT decision
    cadence, ADR-017) -- independent of the training stride STEP_SIZE=32.
    The 64-sample causal window and model weights are unchanged; only how
    often a forward pass fires changes. Training data/labels are untouched
    (this function never writes to data/ or re-trains anything).

    k/w: if None (default), DERIVED from median_span_ms/majority_span_ms so
    the causal post-proc's real-time span stays ~constant across infer_step
    (ADR-017 resolution) -- see derive_kw(). Pass explicit values to
    override (e.g. the old fixed k=5/w=7 for comparison).

    refractory_ms: passed to a RASCueEngine (edge/ras_cue.py, reused
    unchanged) used to compute the WITH-refractory trigger count alongside
    the raw rising-edge count."""
    derived_k, derived_w, decision_interval_ms, median_span_actual_ms, majority_span_actual_ms = \
        derive_kw(infer_step, median_span_ms, majority_span_ms)
    k = k if k is not None else derived_k
    w = w if w is not None else derived_w

    model = load_model()
    scaler = build_deployment_scaler()
    X_raw, y_raw, src_file = load_replay_samples(subject)
    X_scaled = scaler.transform(X_raw)

    # PREDICTION-task ground truth for the sanity check ONLY (never fed to
    # the model -- the model only ever sees X_scaled, one sample at a time).
    y_pred_labels, exclude_mask, onset_idx = FoGPreprocessor.make_prediction_labels(y_raw, HORIZON)

    ring = deque(maxlen=WINDOW_SIZE)
    step_counter = 0
    probs_at_windows = []   # one entry per forward pass (decision-indexed)
    window_end_idx = []     # raw sample index of each forward pass -- for episode_level_metrics reuse

    with torch.no_grad():
        for i, sample in enumerate(X_scaled):
            ring.append(sample)          # CAUSAL: only past+current samples ever enter the ring
            step_counter += 1
            if len(ring) < WINDOW_SIZE or step_counter < infer_step:
                continue
            step_counter = 0

            window = np.stack(ring, axis=0)                          # (64, 3), oldest..newest
            x = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0)  # (1, 3, 64)
            logit = model(x)
            prob = torch.sigmoid(logit).item()
            probs_at_windows.append(prob)
            window_end_idx.append(i)   # this window's last raw sample is i -- CAUSAL, no future index used

    probs_at_windows = np.array(probs_at_windows)
    window_end_idx = np.array(window_end_idx)

    # Same causal post-proc as training (src/run_loso.py), reused unchanged --
    # NOT reinvented. Operates over the DECISION-indexed prob sequence, exactly
    # as the offline evaluation did over its WINDOW-indexed sequence (k=5 means
    # 5 DECISION steps of causal history = 5*infer_step raw samples, which
    # shrinks in real time as infer_step shrinks -- see decision_interval_ms /
    # postproc span reporting below).
    smoothed = causal_median(probs_at_windows, k=k)
    cue_binary = causal_majority_vote((smoothed >= threshold).astype(int), w=w)

    # --- Sanity check 1: discrete cue TRIGGER EVENTS (rising edges of the
    # binary cue stream) -- what a real RAS cueing system would act on. ---
    rising_edges = np.diff(cue_binary, prepend=0) == 1
    n_triggers = int(rising_edges.sum())

    # --- Sanity check 2: reuse run_loso.episode_level_metrics() UNCHANGED --
    # same function, same semantics as the offline evaluation (RUN_20260703_
    # 173337), just fed this stream's window_end_idx/cue_binary instead of
    # the offline batch windows. file_id_all is all-zeros (single replay
    # file). ---
    fake_test_records = [{"onset_idx": onset_idx}]
    file_id_all = np.zeros(len(window_end_idx), dtype=int)
    recalls, lead_times_s, n_skipped = episode_level_metrics(
        fake_test_records, cue_binary, window_end_idx, file_id_all, HORIZON, fs=FS_HZ)

    n_onsets = len(onset_idx)
    n_hits = int(sum(recalls))
    recall_sanity = (n_hits / len(recalls)) if recalls else float("nan")
    mean_lead_sanity = (sum(lead_times_s) / len(lead_times_s)) if lead_times_s else float("nan")

    # Decision cadence + honest post-proc span reporting (ADR-017): k/w are
    # now DERIVED (unless overridden) so their real-time reach stays close
    # to the TARGET median_span_ms/majority_span_ms across infer_step
    # settings. Recomputed here from the FINAL k/w (post explicit-override,
    # if any) -- distinct from the target parameters, which are kept intact
    # for the print statement below (no shadowing).
    decision_interval_ms = (infer_step / FS_HZ) * 1000.0
    actual_median_span_ms = k * decision_interval_ms
    actual_majority_span_ms = w * decision_interval_ms
    combined_postproc_span_ms = (k + w - 1) * decision_interval_ms  # worst-case combined lookback

    # --- Refractory/debounce trigger count (ADR-017 extension): reuse
    # RASCueEngine (edge/ras_cue.py) UNCHANGED, feeding it this stream's
    # cue_binary + window_end_idx (exact raw sample indices, no rate-based
    # timestamp arithmetic needed). Reports the WITH-refractory count
    # alongside the raw rising-edge count (n_triggers) above. ---
    refractory_engine = RASCueEngine(fs_hz=FS_HZ, refractory_ms=refractory_ms)
    refractory_events = refractory_engine.process_stream(
        cue_binary, sample_indices=window_end_idx, verbose=False)
    n_triggers_with_refractory = len(refractory_events)
    n_suppressed_by_refractory = refractory_engine.n_suppressed

    if verbose:
        print("[TASK] PREDICTION (horizon=1.0s pre-FoG, ADR-014) -- NOT detection")
        print(f"[STREAM] subject={subject} src_file={src_file}")
        print(f"[STREAM] n_raw_samples={len(X_scaled)} n_forward_passes={len(probs_at_windows)} "
              f"(one every infer_step={infer_step} samples once the {WINDOW_SIZE}-sample ring fills; "
              f"decoupled from training STEP_SIZE={STEP_SIZE}, ADR-017)")
        print(f"[STREAM] decision_interval_ms={decision_interval_ms:.4f} "
              f"(= infer_step/{FS_HZ:.0f}*1000)")
        print(f"[STREAM] SPAN-CONSTANT post-proc (ADR-017): k={k} "
              f"(target median_span_ms={median_span_ms:.2f}, actual={actual_median_span_ms:.2f}ms), "
              f"w={w} (target majority_span_ms={majority_span_ms:.2f}, actual="
              f"{actual_majority_span_ms:.2f}ms), combined worst-case lookback="
              f"{combined_postproc_span_ms:.2f}ms -- frozen threshold={threshold:.6f} "
              f"(S01-fold val threshold, unchanged from training)")
        print(f"[STREAM] cue trigger events RAW (rising edges of binary cue stream): {n_triggers}")
        print(f"[STREAM] cue trigger events WITH refractory_ms={refractory_ms:.0f}: "
              f"{n_triggers_with_refractory} ({n_suppressed_by_refractory} suppressed)")
        print(f"[STREAM] onsets in replay: {n_onsets} (skipped_no_window={n_skipped})")
        print(f"[STREAM] onsets preceded by >=1 cue within the 1.0s horizon: {n_hits}/{len(recalls)} "
              f"(sanity recall={recall_sanity:.4f})")
        print(f"[STREAM] mean lead time (hit onsets only): {mean_lead_sanity:.4f}s")
        print(f"[STREAM] offline reference for this exact fold+subject (RUN_20260703_173337, "
              f"S01, held-out): episode_recall={OFFLINE_EPISODE_RECALL_S01:.4f}, "
              f"mean_lead_time_s={OFFLINE_MEAN_LEAD_TIME_S01:.4f}")
        print("[STREAM] This is a CORRECTNESS check (does the online ring-buffer + causal "
              "post-proc reproduce the offline evaluation), not a new metric. Small deltas vs "
              "the offline numbers are expected: the offline evaluation DROPS windows that "
              "overlap an excluded during-freeze period entirely (ADR-014); this streaming "
              "replay keeps running continuously through freeze periods too, as a real system "
              "must, so its cue stream can differ near freeze boundaries.")

    return {
        "infer_step": infer_step, "decision_interval_ms": decision_interval_ms,
        "k": k, "w": w,
        "median_span_ms": actual_median_span_ms, "majority_vote_span_ms": actual_majority_span_ms,
        "combined_postproc_span_ms": combined_postproc_span_ms,
        "n_forward_passes": len(probs_at_windows), "n_triggers": n_triggers,
        "n_triggers_with_refractory": n_triggers_with_refractory,
        "n_suppressed_by_refractory": n_suppressed_by_refractory, "refractory_ms": refractory_ms,
        "n_onsets": n_onsets, "n_hits": n_hits, "n_episodes_evaluated": len(recalls),
        "recall_sanity": recall_sanity, "mean_lead_sanity": mean_lead_sanity,
        "cue_binary": cue_binary, "window_end_idx": window_end_idx,
        # cue_binary is DECISION-indexed (one entry every infer_step raw
        # samples), NOT raw-sample-indexed at FS_HZ -- its effective rate is
        # FS_HZ/infer_step. Pass THIS to RASCueEngine(fs_hz=...), not FS_HZ,
        # or timestamps will be wrong by a factor of infer_step (see
        # edge/ras_cue.py docstring) -- or better, pass window_end_idx as
        # `sample_indices` and skip the rate entirely (exact, no factor to
        # get wrong).
        "cue_stream_fs_hz": FS_HZ / infer_step,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=DEFAULT_REPLAY_SUBJECT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_BEST_THRESHOLD)
    parser.add_argument("--k", type=int, default=None,
                         help="Override the derived median-filter k (default: derive from "
                              "--median_span_ms, ADR-017).")
    parser.add_argument("--w", type=int, default=None,
                         help="Override the derived majority-vote w (default: derive from "
                              "--majority_span_ms, ADR-017).")
    parser.add_argument("--infer_step", type=int, default=8,
                         help="Samples between forward passes (deployment decision cadence, "
                              "ADR-017); independent of training STEP_SIZE=32.")
    parser.add_argument("--median_span_ms", type=float, default=800.0,
                         help="Target real-time span (ms) for causal_median -- k is derived to hit this.")
    parser.add_argument("--majority_span_ms", type=float, default=1000.0,
                         help="Target real-time span (ms) for causal_majority_vote -- w is derived to hit this.")
    parser.add_argument("--refractory_ms", type=float, default=2000.0,
                         help="Minimum gap between cue events (edge/ras_cue.py RASCueEngine); 0 disables.")
    args = parser.parse_args()
    replay_stream(subject=args.subject, threshold=args.threshold, k=args.k, w=args.w,
                  infer_step=args.infer_step, median_span_ms=args.median_span_ms,
                  majority_span_ms=args.majority_span_ms, refractory_ms=args.refractory_ms)
