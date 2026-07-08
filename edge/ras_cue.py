"""RAS (Rhythmic Auditory Stimulation) cueing stub -- ADR-016.

Consumes the binary cue signal produced by edge/stream_infer.py and emits
cue EVENTS (rising edges of that stream). No audio hardware tonight -- an
"emitted" event just means logged with a timestamp; wiring real audio output
happens on Jetson.

INTERFACE (documented for the future adaptive variant): CueStrategy is an
abstract base with one method, trigger(event) -> list[beep offsets in
seconds]. FixedTempoCueStrategy is the only one implemented tonight.
AdaptivePhaseShiftCueStrategy is a documented stub that will adjust tempo/
phase to the patient's live-measured gait cadence instead of a fixed tempo --
it slots into RASCueEngine via the same interface, with no changes needed to
RASCueEngine or stream_infer.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CueEvent:
    sample_index: int
    timestamp_s: float


class CueStrategy(ABC):
    """Base interface: given a CueEvent, decide what auditory cue to emit.
    Implementations return a list of beep offsets (seconds, relative to the
    event) -- the caller (RASCueEngine) is responsible for actually
    scheduling/playing them once real audio hardware exists."""

    @abstractmethod
    def trigger(self, event: CueEvent) -> list:
        raise NotImplementedError


class FixedTempoCueStrategy(CueStrategy):
    """Emits a fixed-tempo beep train once triggered -- the only strategy
    implemented tonight. tempo_hz=1.8 (~108 bpm) is a PLACEHOLDER matching a
    typical RAS gait-cueing cadence from the literature; it is NOT tuned to
    any patient -- a config knob for future work (see DECISIONS.md ADR-016)."""

    def __init__(self, tempo_hz=1.8, duration_s=2.0):
        self.tempo_hz = tempo_hz
        self.duration_s = duration_s

    def trigger(self, event: CueEvent) -> list:
        period = 1.0 / self.tempo_hz
        n_beeps = int(self.duration_s / period) + 1
        return [round(i * period, 4) for i in range(n_beeps)]


class AdaptivePhaseShiftCueStrategy(CueStrategy):
    """STUB -- NOT implemented tonight. Would adjust beep tempo/phase to the
    patient's live-measured gait cadence (e.g. stride-frequency estimated
    from the IMU stream) instead of a fixed tempo, to entrain gait rhythm
    rather than just alert. Slots in via the same CueStrategy interface --
    RASCueEngine and stream_infer.py do not need to change when this is
    implemented."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AdaptivePhaseShiftCueStrategy is a documented interface stub, "
            "not implemented yet -- see DECISIONS.md ADR-016.")

    def trigger(self, event: CueEvent) -> list:
        raise NotImplementedError


class RASCueEngine:
    """Consumes a binary cue stream (as produced by edge/stream_infer.py),
    detects rising edges as discrete CueEvents, and dispatches each to a
    CueStrategy. No audio hardware -- 'emit' means log for now.

    IMPORTANT: cue_binary from stream_infer.py is DECISION-indexed (one
    entry every infer_step raw samples, ADR-017), NOT raw-sample-indexed at
    FS_HZ=64. Pass the true per-entry RAW sample index via `sample_indices`
    (stream_infer.py returns this as window_end_idx) so event timestamps are
    correct -- if `sample_indices` is omitted, timestamps fall back to
    `i / fs_hz`, which is only correct if cue_binary really is
    raw-sample-indexed (true for the __main__ synthetic demo below, NOT
    true for stream_infer.py's output).

    REFRACTORY / DEBOUNCE (ADR-017 extension): after a cue event fires, any
    rising edge within `refractory_ms` of it is SUPPRESSED (not emitted as a
    new event) -- a patient should not be re-cued every 125ms just because
    the underlying binary stream flickers. Default 2000ms. Set to 0 to
    disable (raw rising-edge behavior, e.g. to reproduce pre-refractory
    counts for comparison)."""

    def __init__(self, strategy: CueStrategy = None, fs_hz=64.0, refractory_ms=2000.0):
        self.strategy = strategy or FixedTempoCueStrategy()
        self.fs_hz = fs_hz
        self.refractory_ms = refractory_ms
        self.events = []
        self.n_suppressed = 0

    def process_stream(self, cue_binary, sample_indices=None, verbose=True):
        prev = 0
        last_event_t = None
        n_suppressed = 0
        for i, v in enumerate(cue_binary):
            if v == 1 and prev == 0:
                raw_idx = int(sample_indices[i]) if sample_indices is not None else i
                t = raw_idx / self.fs_hz
                if last_event_t is not None and (t - last_event_t) * 1000.0 < self.refractory_ms:
                    n_suppressed += 1   # rising edge occurred, but within the refractory window
                    prev = v
                    continue
                event = CueEvent(sample_index=raw_idx, timestamp_s=t)
                beeps = self.strategy.trigger(event)
                self.events.append(event)
                last_event_t = t
                if verbose:
                    dur = getattr(self.strategy, "duration_s", float("nan"))
                    print(f"[RAS_CUE] event @ t={event.timestamp_s:.3f}s (sample {raw_idx}) -- "
                          f"{self.strategy.__class__.__name__}, {len(beeps)} beeps "
                          f"scheduled over {dur:.2f}s")
            prev = v
        self.n_suppressed = n_suppressed
        return self.events


if __name__ == "__main__":
    import numpy as np
    demo = np.zeros(200, dtype=int)
    demo[50:55] = 1
    demo[120:123] = 1
    engine = RASCueEngine()
    events = engine.process_stream(demo)
    print(f"[RAS_CUE] demo: {len(events)} cue events detected from a synthetic stream")
