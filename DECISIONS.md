# DECISIONS.md — Architecture/Methodology Decision Records (ADR)

Each entry: the decision, the alternatives considered, and WHY. This file is the
defense script for the 推甄 exhibition and any conference review. When a decision
changes, append a new ADR that supersedes the old one — do not edit history.

Status legend: ACCEPTED · PROPOSED · TO-FIX (known issue, not yet fixed) · OPEN
(needs a fact/input before deciding) · SUPERSEDED.

---

## ADR-001 — Subject-level LOSO isolation · ACCEPTED
Outer loop = Leave-One-Subject-Out over the FoG subjects; no window from a test
subject ever appears in train/val. Alternative: random window split (rejected —
inflates metrics via near-duplicate adjacent windows from the same person, which
is scientifically invalid for a patient-independent claim).

## ADR-002 — Exclude S04 & S10 from LOSO; use as isolated False-Alarm test · ACCEPTED
S04 and S10 contain no FoG events (single-label-0); confirmed in the data
(loso_summary shows NaN AUC/F1 for both — no positive class). Keeping them in the
LOSO loop dilutes training and produces undefined per-subject F1/AUC. Decision:
route them to a separate False-Alarm-Rate (FAR) suite that measures how often the
model fires on patients who never freeze. This mirrors the standard practice of
holding out non-freezers to estimate specificity in the wild.

## ADR-003 — Threshold selection: move from test to validation · TO-FIX
EVIDENCE: in loso_summary_20260519, Best_Threshold differs per test subject
(0.24–0.37) and Best_F1 is the max-F1 swept over thresholds on that test subject.
That is test-set tuning = optimistic bias. The reported mean Best-F1 (~0.50) and
the "+4% from post-processing" both inherit this bias.
DECISION: choose the threshold on the VALIDATION subject, freeze it, apply to the
TEST subject. Report F1 at that fixed threshold as the headline; keep swept-Best-F1
only as a labeled upper bound. Re-run the post-processing comparison under this
rule before claiming the +4%.

## ADR-004 — Post-processing must be causal for deployable metrics · TO-FIX
Median filter (K=5) + rolling majority vote (W=7) are currently (assumed) centered,
i.e. they peek at future samples — invalid for a real-time predictor and a source
of inflated F1. DECISION: implement causal (past-only) variants for any number
claimed as real-time; report end-to-end latency including the post-proc buffer
wait, not just model forward time. Centered versions may stay only for offline
analysis plots, clearly labeled.

## ADR-005 — Receptive-field check must match block structure · OPEN
RF = 1 + n_conv_per_block * sum((K-1) * dilation_l). The single-conv formula
under-estimates RF by ~2x if basic_tcn uses the standard 2-conv residual block.
BLOCKER: confirm how many dilated convs per residual block in basic_tcn.py.
Until confirmed, treat the grid combo (K=7, dilations=[1,4,8]) as suspect (single-
conv RF = 79 > 64 already; two-conv = 157).

## ADR-006 — Reframe detection -> prediction via label-shift, horizon = 1 s · ACCEPTED (planned)
Daphnet has no pre-FoG label, so we synthesize it: relabel the H seconds before
each FoG onset as the positive (pre-FoG) class. Target horizon H = 1 s (64 samples
at 64 Hz), consistent with published FoG-prediction lead times. Detection remains
the fallback model for the closed-loop demo if prediction is unreliable.

## ADR-007 — Metrics suite · ACCEPTED
Report, per subject and averaged: ROC-AUC AND PR-AUC (AP) [imbalance ~81/19], F1
at the frozen threshold, episode-level detection rate, and detection latency.
Sample-wise F1 alone is not clinically meaningful.

## ADR-008a — Dataset strategy: keep Daphnet primary + add external validation · PROPOSED
Rather than rebuild on a new dataset under time pressure, keep Daphnet as
train/dev (pipeline works; ankle+thigh matches the planned deployment) and add a
second public dataset (DeFOG or O'Day et al. 2022 open data) as a CROSS-DATASET
external test set ("train on Daphnet, test on unseen dataset"). This buys
conference-grade generalization evidence without a destructive rebuild. Full
switch to DeFOG/tDCS is the higher-ceiling but higher-risk alternative; it also
forces a lower-back sensor, conflicting with ankle deployment.

## ADR-008b — Staged dataset migration via data-spike · SUPERSEDES ADR-008
Hardware is not yet purchased, so sensor placement can match any dataset; Daphnet's
small size genuinely caps the generalization claim. Decision: (1) fix eval protocol
(ADR-003/004/005) first; (2) build a leak-free nested auto-tuning workflow (Optuna,
TPE+pruning) — reusable across datasets; (3) run a 3–5 day ingestion spike on a few
DeFOG subjects; (4) if labeling/IO is tractable, promote DeFOG to primary and demote
Daphnet to an external cross-dataset TEST set; else keep Daphnet primary + DeFOG as
external test. Do NOT commit 10 weeks to a switch before the spike. Auto-tuning lowers
model-rebuild cost but NOT data-plumbing cost, which dominates a switch.

## ADR-009 — IMU: single ankle, 6-axis, >=64 Hz · PROPOSED
O'Day et al. 2022 report single-ankle AUROC 0.80 vs three-IMU 0.83 — extra sensors
give marginal gain. Decision: buy ONE ankle IMU (accel+gyro), I2C, sample at
>=64 Hz to align with Daphnet. Sensor placement is deliberately coupled to the
dataset choice in ADR-008 (ankle <-> Daphnet/O'Day).

## ADR-010 — Edge target: Jetson Orin NX 8 GB, ONNX -> TensorRT · PROPOSED
Export PyTorch -> ONNX -> TensorRT engine. Start FP16; pursue INT8 quantization
later to lower the compute threshold / enable a smaller device. Streaming inference
uses a ring buffer; the causal post-proc (ADR-004) runs on the stream. Validate
on-device output parity against offline results on replayed Daphnet before any
live IMU test.

---
### Open blockers feeding these ADRs
- ADR-005: n_conv_per_block in basic_tcn.py.
- ADR-003/004: exact location of threshold + post-proc code in run_tcn_v2.py.
