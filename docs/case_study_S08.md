# Case Study: S08 — Below-Chance AUC as an Akinetic-Freezing Generalization Failure

**Status:** CLOSED (diagnosed, not fixed — kept as-is per decision below)
**Task:** DETECTION baseline, `RUN_20260702_172209` (leak-free protocol, ADR-003/004/005)
**Related files:** `results/reports/03_case_studies/diag_S08_report.txt`, `results/reports/03_case_studies/diag_S08/`,
`scripts/diagnose_subject.py`, `results/reports/01_baselines/baseline_two_views.txt`

## The numbers

| Metric | S08 | Training pool (excl. S08) |
|---|---|---|
| Test_ROC_AUC | 0.3599 (below chance) | mean 0.7770 (7-fold, excl. S08) |
| Test_F1 | 0.0000 | mean 0.3370 (7-fold, excl. S08) |
| Positive-window fraction | 0.28 | pool mean 0.135, std 0.072 (z = 2.02) |
| FoG events / total duration | 14 events / 200.9 s (~14.4 s/event) | — |

S08 is a single-file subject (`S08R01.txt`); its below-chance AUC single-handedly pulls the
official 8-fold headline down by ~0.05 ROC-AUC and ~0.04 F1 versus the 7-fold sensitivity view
(see `results/reports/01_baselines/baseline_two_views.txt` for the full with/excl. comparison).

## Why this is not a data defect

Three independent checks argue against a corrupted label file or a mis-oriented/mis-scaled
sensor for S08:

1. **No axis sign flip.** The gravity-bearing ankle_y axis mean is 981.7 mg for S08 vs 989.5
   (S05, healthy reference) and 1011.9 (training pool) — same sign, same magnitude. A
   mounted-backwards or axis-swapped IMU would most plausibly show a sign flip or large offset
   on this axis; none is present. ankle_x and ankle_z are likewise same-sign and within one
   pool standard deviation of the pool mean.
2. **No clean reversal in the predicted-probability distributions.** If the labels or sensor
   axis were inverted, the model's predicted probabilities for the true-positive and
   true-negative windows would form two well-separated clusters with the wrong one on top.
   Instead, S08's label=0 and label=1 distributions sit in the same narrow low-probability
   band with heavy overlap, and label=1 is only marginally shifted the wrong way (mean
   probability delta = -0.013). The healthy reference S05 shows the same weak-overlap
   pattern, just correctly directioned. See the histogram comparison:
   [`results/reports/03_case_studies/diag_S08/prob_histogram_S08.png`](../results/reports/03_case_studies/diag_S08/prob_histogram_S08.png)
   vs
   [`results/reports/03_case_studies/diag_S08/prob_histogram_S05.png`](../results/reports/03_case_studies/diag_S08/prob_histogram_S05.png).
3. **A plausible akinetic onset in the raw signal.** Around S08's first labeled FoG onset, the
   raw ankle-accelerometer signal visibly quiets down (drops in amplitude/variance) rather than
   spiking — consistent with an *akinetic* (motor-block) freezing episode, a recognized FoG
   phenotype distinct from high-frequency leg-trembling freezing. See:
   [`results/reports/03_case_studies/diag_S08/raw_segment_S08_vs_S05.png`](../results/reports/03_case_studies/diag_S08/raw_segment_S08_vs_S05.png).

Raw label composition is also physiologically ordinary: 14 discrete FoG events over ~201 s,
averaging ~14 s/event, in the same range as S05's own events (66 events / 474 s, ~7 s/event) —
no impossible run lengths or scrambled labeling.

## Mechanism

The most defensible explanation: the model is trained predominantly on subjects whose FoG
episodes show **high-variance / trembling** accelerometer signatures, and appears to have
implicitly learned a heuristic close to "increased signal variance -> FoG." For S08, whose FoG
episodes are comparatively **quiet (akinetic)**, that heuristic fires backwards — the model
assigns *lower* probability exactly when S08 is freezing. This produces a weak but temporally
consistent anti-correlation (confirmed by causal-median smoothing pushing the AUC further below
0.5, from 0.405 raw to 0.360 smoothed, rather than regressing toward 0.5 as pure i.i.d. noise
would) without requiring any inverted label or flipped sensor axis. This is a **class-conditional
domain-generalization failure**, not a data-quality defect.

## Decision

Keep S08 in the LOSO pool; report its below-chance AUC honestly in the headline mean per
CLAUDE.md S6 (metrics under imbalance must not be cherry-picked to inflate the reported number).
Treat it as a documented case study rather than an exclusion candidate — see the corresponding
ADR in `DECISIONS.md`.

## Implication for the PREDICTION phase

If detection already assigns *lower* confidence during an akinetic freeze than during normal
gait, predicting that same freeze *before* it starts (horizon = 1 s, ADR-006) will be at least as
hard: an akinetic onset offers less of the high-variance lead-in signal the model currently
relies on, so pre-freeze windows for this phenotype are likely to be even less distinguishable
from normal gait than the freeze window itself. Akinetic-phenotype subjects should be watched
specifically when the prediction-horizon work starts.

## PREDICTION-task result (2026-07-03 update)

The full 8-fold PREDICTION baseline (RUN_20260703_173337, horizon=1s, ADR-014)
is now available, and it **falsifies the prediction above** for S08:

| Metric | S08 under DETECTION (RUN_20260702_202440) | S08 under PREDICTION (RUN_20260703_173337) |
|---|---|---|
| Test_ROC_AUC | 0.4908 (below chance) | 0.666 |
| episode_recall | n/a (detection has no episode metric) | 0.714 (10/14 onsets) |
| mean lead time | n/a | 0.836 s (of the 1.00 s horizon — the LONGEST of any of the 8 PREDICTION folds) |

Rather than being *at least as hard* as detection, PREDICTION is the ONE task
where S08 is not below-chance. See ADR-015 for the full interpretation; the
short version: DETECTION's "high signal-variance -> FoG" heuristic (the
mechanism diagnosed above) simply never gets a chance to fail on S08 during
PREDICTION, because PREDICTION only ever looks at the window strictly
*before* onset, never the quiet in-freeze window itself. That pre-onset
window apparently carries its own distinguishable signal (plausibly a
motor-preparation or gait-adaptation cue, distinct from the akinesia that
confuses DETECTION) that the PREDICTION model picks up despite never having
learned DETECTION's variance heuristic. This is genuine evidence that
**DETECTION and PREDICTION are sensitive to different parts of the movement
signal**, not merely that one task is uniformly harder than the other — a
stronger and more specific finding than "S08 is a hard subject," worth
carrying into the write-up as its own point (ADR-015).

Caveat: this is n=1 subject and PREDICTION's episode counts are small
(10-64 per fold across the 8-fold run); not yet confirmed as a general
pattern across other akinetic-phenotype subjects, since S08 is currently
the only one identified in Daphnet.
