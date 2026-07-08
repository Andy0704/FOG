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

## ADR-003 — Threshold selection: move from test to validation · RESOLVED
run_loso.py now does a nested train/val/test subject split per fold; best_thresh
is chosen on val_subj (Youden's J on causal-median-smoothed probs) and frozen
before being applied to test_subj. Headline metric is Test_F1 at that frozen
threshold (see ADR-004/ADR-007 for post-proc + full metric suite).

**val_subj selection TODO -- FULLY RESOLVED 2026-07-03 (Option 1: rotate +
median-select, validated at full scale).**
The old code fixed `val_subj = pool[0]`, which meant 6 of 8 outer folds all
validated on S01 -- biasing both early stopping and threshold selection toward
whatever S01 happens to look like. run_loso.py now does a genuine inner LOSO
per outer fold: for each of the 7 remaining subjects as a val-candidate, train
a fresh model on the other 6 and score it by ROC-AUC on causal-median(k=5)-
smoothed val probs (a metric distinct from the per-epoch early-stopping
criterion, which stays val focal-loss, unchanged in trainer_tcn.py). The
val_subj whose score is the MEDIAN of the 7 (not the max) is selected --
median avoids picking whichever candidate got luckiest, which the max would
do. The model is then retrained once on that split (train = pool minus
selected val_subj) for the actual TEST evaluation, keeping one test-time model
per outer fold as before. Hard leakage constraint (test_subj never in any
train/val list, inner or outer) is asserted at both the inner-candidate level
and the final split level -- no leakage fired in the smoke test or the full
run. OPTION 2 (ensemble the 7 inner-split models' test-set predictions instead
of retraining once) is deliberately NOT implemented -- left as a TODO in
run_loso.py for a future rigor pass, since it would require keeping 7
checkpoints per outer fold instead of discarding them. Cost: ~8x more models
trained per outer fold (7 inner + 1 final retrain vs. 1 before).

Smoke-tested on test=S01, n_epochs=10 (results/reports/05_smoke_tests/innerloso_smoke.txt),
then validated at FULL SCALE: 8-fold x 50-epoch run RUN_20260702_202440
(3708s wall-clock, launched unattended via tmux), full table in
results/reports/01_baselines/baseline_innerloso.txt. The bias fix RAISED the mean and
LOWERED the variance on all three headline metrics vs. the old fixed-val
baseline (RUN_20260702_172209): Test_F1 0.2949±0.2061 -> 0.3664±0.1439
(+0.0715), Test_PR_AUC 0.3719±0.1511 -> 0.4211±0.1081 (+0.0492),
Test_ROC_AUC 0.7249±0.1723 -> 0.7743±0.1261 (+0.0494). inner_scores confirm
the mechanism: S02 was the highest-scoring inner val candidate in every fold
it was eligible for, yet was never selected (0/8) -- exactly the "lucky val"
bias a max-rule would have picked every time; S01 dropped from validating 6/8
folds to 3/8; S08 was the lowest-scoring candidate in every fold, consistent
with its akinetic-phenotype below-chance test result (docs/case_study_S08.md).
RUN_20260702_202440 is now the OFFICIAL reference baseline (PROJECT_STATUS.md
§3); RUN_20260702_172209 is superseded but retained on disk for comparison.

## ADR-004 — Post-processing must be causal for deployable metrics · RESOLVED
run_loso.py now uses `causal_median` (rolling median, min_periods=1, past-only)
and `causal_majority_vote` (rolling mean >= 0.5, past-only) for both threshold
selection (on val) and final test evaluation. No centered post-proc remains in
the metrics path.

## ADR-005 — Receptive-field check must match block structure · RESOLVED
Confirmed: TemporalBlock has 2 dilated convs per residual block. basic_tcn.py now
exposes `receptive_field(kernel_size, dilations, n_conv_per_block=2)` and
BasicTCN.__init__ asserts RF <= window_size before building layers (raises
ValueError otherwise), printing `[RF] <rf> (<= <window_size>)`.

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

## ADR-011 — Training env moved to WSL-native venv (fog_env_wsl) · ACCEPTED
Windows `fog_env_64` required a separate shell from the Claude CLI (WSL), causing
recurring encoding/multi-shell friction. `nvidia-smi` inside WSL already showed the
RTX 4060 (driver 591.86, CUDA 13.1 capable, backward-compatible with cu121), so a
WSL-native venv was viable. Created additively — `fog_env_64` is untouched on disk,
both venvs remain available. Exact install command that reproduced the pinned
torch==2.5.1+cu121:
```
python3 -m venv ./fog_env_wsl
source ./fog_env_wsl/bin/activate
python -m pip install --upgrade pip
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
Verified: `torch.__version__` = `2.5.1+cu121`, `torch.cuda.is_available()` = True,
`torch.cuda.get_device_name(0)` = `NVIDIA GeForce RTX 4060`. `~/.bashrc` now exports
`PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` so CLI + training share one encoding-safe
shell. Side note: `requirements.txt` line 1 had a corrupted entry
(`Test-Path .gitignore contourpy==1.3.3`, a stray pasted PowerShell command merged
with the `contourpy` line) — fixed to `contourpy==1.3.3` since it blocked `pip install
-r requirements.txt`; no other lines touched.

## ADR-012 — Keep S08 in the LOSO baseline; report its below-chance AUC honestly · ACCEPTED
Diagnosis (`scripts/diagnose_subject.py`, `results/reports/03_case_studies/diag_S08_report.txt`,
`docs/case_study_S08.md`) found no sensor-orientation flip and no clean label-reversal
signature for S08 (Test_ROC_AUC=0.3599, Test_F1=0.0 in RUN_20260702_172209); the raw
signal shows a plausible akinetic (quiet) freezing onset instead of a trembling one,
and the model's learned "high variance -> FoG" heuristic fails to transfer to that
phenotype. This is a genuine class-conditional generalization failure, not a data
defect. Decision: keep S08 in the 8-fold headline mean per S6 (do not drop hard
subjects to inflate the reported metric); treat it as a documented case study for the
write-up rather than an exclusion candidate. A 7-fold "excl. S08" number exists only
as a sensitivity view (`results/reports/01_baselines/baseline_two_views.txt`), never as the
headline.

## ADR-013 — Optuna hyperparameter tuning on a VAL-ONLY objective, TPE search · PROPOSED (smoke-tested)
Goal: push the DETECTION baseline (RUN_20260702_202440, inner-LOSO, ROC-AUC
0.7743) via hyperparameter search, without ever letting the tuner see a TEST
subject.

**Objective (val-only, anti-leakage).** A trial's objective is the MEAN of
the per-outer-fold MEDIAN inner-LOSO validation ROC-AUC -- the exact same
median-rule metric run_loso.py already computes for val_subj selection
(ADR-003), aggregated over a REDUCED subset of outer folds (3 of 8: S01,
S02, S03) rather than all 8, for cost control. Test subjects' data is never
loaded during tuning: the objective only ever calls run_split() with
inner_train_subjs and a val candidate drawn from the pool excluding the
outer test_subj; build_xy()/build_prediction_records() are never called with
a test_subj anywhere in the tuning code path. After a study completes, the
best params get ONE final full inner-LOSO run (all 8 outer folds, n_epochs=
50) via run_loso.main() -- reused, not reimplemented -- so that run's test
metrics are the first time the tuned model ever sees a test subject.

**Search space.** kernel_size/dilations restricted to the 4 combos with
RF<=64 (K=3 d=[1,2,4] RF=29; K=3 d=[1,4,8] RF=53; K=3 d=[1,2,8] RF=45;
K=5 d=[1,2,4] RF=57, the current baseline's architecture) -- Optuna picks an
INDEX into this fixed list, so it can never construct an invalid combo; this
is a structural guarantee, not a runtime skip/prune. A defensive RF
recomputation + BasicTCN's own ValueError are also verified to reject a
known-bad combo (K=5, d=[1,4,8], RF=105) in the smoke. Also tuned: dropout
(0.1-0.4), focal alpha (0.5-0.9), focal gamma (1.0-3.0), lr (1e-4..3e-3, log),
weight_decay (0..1e-3), batch_size ({128,256}). num_channels=[32,64,128] kept
fixed for now.

**TPE over grid, why.** The 4-combo architecture axis is enumerable, but
crossed with 6 continuous/categorical hyperparameters a grid becomes
combinatorially expensive for no benefit -- TPE (Tree-structured Parzen
Estimator) samples informedly from prior trial outcomes instead of
exhaustively enumerating, and pairs naturally with MedianPruner to abort
clearly-bad trials early (after fold 1 of 3, once >=5 trials give the
pruner a real median to compare against), which a plain grid search cannot
do. optuna.samplers.TPESampler(seed=42) + optuna.pruners.MedianPruner
(n_startup_trials=5, n_warmup_steps=1).

**Implementation.** New file `src/tune_optuna.py`; reuses (does not
reimplement) `run_loso.py`'s `run_split()` / `inner_val_score()` and
`basic_tcn.py`'s `receptive_field()`. Study persisted to
`results/optuna/study_{smoke,full}.db` (sqlite) + a crash-safe incremental
per-trial-per-fold CSV log (same append-per-row pattern as run_loso.py's
`append_result_row()`).

**Checkpoint-isolation bug found and fixed during the smoke.** The first
smoke attempt crashed with a PyTorch state_dict shape-mismatch: run_split()
(and trainer_tcn.py) saved every run's checkpoint to a single hardcoded
`best_model.pth` in the repo's CWD, and a second, independent Claude Code
session was concurrently training against the same repo at the same time
(PREDICTION-task work, ADR-014) -- both processes wrote to the same shared
filename, and because this smoke varies architecture between calls, the
collision surfaced as a visible crash instead of silently corrupting a
checkpoint. Fixed by giving every `run_split()` call its own unique
checkpoint path (`results/_tmp_checkpoints/ckpt_<pid>_<uuid>.pth`, cleaned up
by the caller after use); `TCNTrainer` gained an optional `checkpoint_path`
param (default unchanged) to support this. This also protects any future
concurrent training runs from either session, not just Optuna's -- a latent
hazard that predates this task but was only exposed by architecture-varying
trials.

**Status.** Smoke-tested (2 trials x 1 outer fold x 5-epoch cap,
`results/reports/02_optuna_tuning/optuna_smoke.txt`): RF guard rejects the known-bad combo,
objective confirmed val-only (test subject never touched, verified against
the trial log), no leakage assertion fired, trials persisted and reloadable
from disk. Full 30-trial study (epoch_cap=25, 3 tuning folds) estimated at
~5.85 hr upper bound (conservative -- assumes no early stopping/pruning
benefit) -- **NOT YET LAUNCHED**, awaiting approval given the cost.

## ADR-014 — PREDICTION task definition: H=1s pre-FoG, freeze excluded, episode-level eval · VALIDATED (full 8-fold)
Reframes ADR-006 (detection -> prediction via label-shift) into a concrete,
implemented label definition, added as a NEW task mode alongside DETECTION
(default; DETECTION behavior is byte-for-byte unchanged when task="detection").

**Definition.** horizon H = 64 samples (1s @ 64Hz). Onset = the index i in a
single continuous PER-FILE (never cross-file/cross-subject concatenated)
0-filtered Daphnet label sequence where label[i-1]==0 (no-freeze) and
label[i]==1 (freeze) -- i.e. the raw Daphnet 1->2 transition, post 0-filtering.
Positive (pre-FoG) class = the H samples strictly before each onset. Every
during-freeze sample (onset included) is marked in an exclude_mask and is
NEVER given a positive/negative label -- we are predicting imminent onset, not
ongoing freeze. A window's label is taken from its END-POINT sample (not
any-overlap, unlike DETECTION); a window is DROPPED entirely (not merely
labeled 0) if ANY sample it covers is in the exclude_mask, so no window mixes
FoG-episode content with a "normal" label. If two onsets are closer than H,
overlapping pre-FoG/exclude regions are resolved in favor of exclusion (it
always wins at window-build time).

**Why per-file, not per-subject/per-pool.** Concatenating raw label arrays
across files (as build_xy() already does for DETECTION) before running onset
detection would fabricate onsets and pre-FoG horizon windows that straddle two
unrelated recordings -- worse, for TRAIN pools spanning multiple subjects, it
would fabricate cross-SUBJECT pre-FoG windows. Implementation
(fog_preprocessing.FoGPreprocessor.make_prediction_labels,
run_loso.build_prediction_records/windows_from_records) always builds labels
file-by-file, concatenating only the already-computed per-file labels/masks
afterward. The scaler is still fit once on the full concatenated TRAIN raw
signal (S1 unaffected), only the label/windowing step is per-file.

**Episode-level evaluation (new).** Per FoG onset in the TEST subject: recall
= was there >=1 positive prediction among windows whose endpoint falls in that
onset's true pre-FoG horizon; lead time = (onset - endpoint of the earliest
hit window) / 64, reported in seconds, averaged over hit episodes only. This
is the primary clinically-relevant metric for this task -- window-level
F1/PR-AUC collapse under the induced class imbalance (see below) in ways that
under-represent whether the model is actually catching onsets early.

**Smoke result (1-fold, test=S01, n_epochs=10, results/reports/05_smoke_tests/prediction_smoke.txt,
RUN_20260703_160327):** pipeline verified correct (RF=57<=64, no leakage,
exclude_mask removes 13.99%/5.50% of samples train-pool/test respectively,
onsets_detected=214/23). Positive (pre-FoG) window rate is ~1-4% across every
inner-LOSO candidate and TEST=S01 (vs DETECTION's ~19%, Daphnet's native
81/19 split per S6) -- a further 5-15x sparsification, an intrinsic
consequence of H=window_size=64 combined with step_size=32 (each onset
contributes at most ~2 positive windows by construction), not a
preprocessing defect. Window-level Test_F1 dropped -90.2% (0.3487 -> 0.0343)
and Test_PR_AUC -92.7% (0.4032 -> 0.0293) vs the DETECTION baseline for S01
(RUN_20260702_202440); Test_ROC_AUC dropped a much smaller -16.8% (0.8774 ->
0.7304). Episode-level recall = 0.7826 (18/23 onsets got >=1 correct
imminent-onset window) with mean lead time 0.6970s of the 1.00s horizon --
materially more encouraging than the window-level numbers alone suggest.
Full 8-fold PREDICTION run intentionally NOT launched pending review of this
smoke.

**FULL 8-fold result (n_epochs=50, all 8 FOG subjects, results/reports/
baseline_prediction.txt, RUN_20260703_173337, 2026-07-03):** ran as a pure
execution of this already-validated pipeline -- NO code changes, inner-LOSO
val_subj selection reused unchanged. Zero leakage-assertion failures across
all 64 models trained (8 outer folds x 8 splits). Total wall-clock 3664.1s
(~61.1 min), well under the ~2.55hr no-early-stopping projection.

**Episode-level recall/lead-time is hereby established as the PRIMARY
reported metric for the PREDICTION task** (not window-level F1/PR-AUC):
mean episode_recall = 0.6981 +/- 0.1606 (pooled across all 233 onsets =
171/233 = 0.7339), mean lead time = 0.7154 +/- 0.0700s of the 1.00s horizon,
mean Test_ROC_AUC = 0.7404 +/- 0.0858. Window-level Test_F1 (0.0939 +/-
0.0636) and Test_PR_AUC (0.0934 +/- 0.0580) remain low across all 8 folds --
confirmed as a SPARSITY ARTIFACT of the ~1-4% pre-FoG positive rate (range
0.54%-4.06% across folds), not a signal-quality problem: ROC-AUC (rank-based,
not threshold/imbalance-sensitive) stays comparable in absolute terms to
several DETECTION folds, and episode-level recall shows the model reliably
catches most onsets with substantial lead time when evaluated at the
event level rather than the window level. 1 fold flagged for low episode
recall (S09=0.4074, 11/27, though its window-level PR-AUC/F1 were near-best
of all 8 folds -- a specific missed-onset pattern worth a follow-up look,
not a general signal failure). S06 is the sparsest fold (n_pos=20, only 10
onsets) but not below the n_pos<20 flag threshold.

**S08 (akinetic phenotype) finding:** below-chance under DETECTION
(Test_ROC_AUC=0.4908, ADR-012) but NOT below chance under PREDICTION
(Test_ROC_AUC=0.6659, episode_recall=0.7143, and the LONGEST mean lead time
of any fold at 0.8359s). This is genuinely informative for the write-up: the
pre-onset window carries a distinguishable kinematic signature for this
subject that DETECTION's "high-variance -> FoG" heuristic does not pick up
during the quiet/akinetic freeze itself -- the two tasks are evidently
sensitive to different parts of the movement signal (pre-onset preparation
vs. during-freeze tremor/akinesia), which is a stronger, more specific
finding than "S08 is just a hard subject."

## ADR-015 — Cross-task finding: DETECTION and PREDICTION read different parts of the movement signal · ACCEPTED
**Decision: document and keep this as a standalone, citable finding** (not
just a footnote inside ADR-014), since it independently corroborates the
S08 diagnosis (ADR-012, docs/case_study_S08.md) and is a genuine scientific
result for the write-up, not just an engineering note.

**The finding.** S08 (the akinetic/quiet freezer, ADR-012) is below chance
under DETECTION (Test_ROC_AUC=0.4908, RUN_20260702_202440) but is NOT below
chance under PREDICTION (Test_ROC_AUC=0.6659, episode_recall=0.7143, and the
LONGEST mean lead time of any of the 8 PREDICTION folds at 0.8359s,
RUN_20260703_173337). Full numbers: docs/case_study_S08.md ("PREDICTION-task
result" section) and results/reports/01_baselines/baseline_prediction.txt section 5.

**Interpretation.** ADR-012 diagnosed DETECTION's S08 failure as a learned
"high signal-variance -> FoG" heuristic that fails to transfer to S08's
comparatively quiet in-freeze accelerometer signature. If that heuristic
were the ONLY signal available anywhere in the trace, PREDICTION -- which
only sees the H=1s window strictly BEFORE onset, never the freeze itself --
should do no better, and docs/case_study_S08.md's original "Implication for
the PREDICTION phase" section (written before this data existed) explicitly
predicted PREDICTION would be *at least as hard* for akinetic onsets. The
full 8-fold PREDICTION run falsifies that specific prediction for S08: the
pre-onset window apparently carries a distinguishable kinematic signature
(likely a motor-preparation / gait-adaptation signal distinct from the
in-freeze akinesia itself) that DETECTION's variance-based heuristic never
learns to use, because DETECTION is never trained to look at that window in
isolation. This is evidence the two task framings are sensitive to
genuinely different parts of the movement signal, not just "the same signal
at different difficulty," and is a stronger, more specific claim for the
write-up than "S08 is a hard subject" or "PREDICTION is harder than
DETECTION" in general.

**Caveat.** n=1 subject (S08) showing this reversal; not yet confirmed as a
general pattern across other subjects/phenotypes, and PREDICTION's episode
counts per fold are small (10-64, section 4 of results/reports/
baseline_prediction.txt) so individual-fold ROC-AUC has real sampling
noise. Treat as a documented case-study finding worth a figure/paragraph in
the write-up, not yet a mechanistic claim to build architecture decisions
on without further validation (e.g. cross-checking against DeFOG/O'Day if
that dataset is reached, ADR-008b).

## ADR-016 — Deployment plan: export -> stream -> cue -> latency, hardware-independent parts first · ACCEPTED
**Decision.** Build the Jetson deployment pipeline in four independent
stages -- ONNX export, streaming inference, RAS cueing, latency measurement
-- and do every stage that does NOT require physical Jetson/IMU hardware
FIRST, on WSL/RTX4060, before any TensorRT conversion or real-sensor wiring.
New package `edge/` (was empty), self-contained, importing READ-ONLY from
`src/` (never modifying training logic): `model_utils.py` (shared config/
checkpoint/scaler loading), `export_onnx.py`, `stream_infer.py`,
`ras_cue.py`, `latency_harness.py`.

**Why pipeline-first, before final model tuning.** The trained model
(currently RUN_20260703_173337's PREDICTION checkpoints) is a SWAPPABLE
ARTIFACT in this pipeline -- Optuna tuning (ADR-013) or a future full-data
retrain can drop in a new checkpoint without touching export/stream/cue/
latency code. Conversely, the deployment *pipeline itself* (ring-buffer
causal windowing, ONNX export shape/opset, TensorRT compatibility, cueing
event structure, latency measurement methodology) has NOTHING to do with
which specific checkpoint is loaded, and its risks (dependency
availability, opset/TensorRT compatibility, causal correctness of the
streaming post-proc, timestamp bugs at integration boundaries) are
ENGINEERING risks that are cheaper to surface now than the night before a
demo. Latency in particular can ONLY be honestly measured on real
target hardware (Jetson) -- there is no shortcut around that -- so the
right split of tonight's effort vs tomorrow's is: everything that is
genuinely hardware-independent (export code, streaming logic, cueing
interface, the latency HARNESS itself) done and smoke-tested tonight;
everything that requires the physical device (TensorRT engine build, real
IMU input, the real latency NUMBERS) explicitly deferred, not faked.

**Deployment checkpoint choice.** Default = the S01-fold checkpoint of
RUN_20260703_173337 (`best_model_S01.pth`, trained on
train_subjs=[S03,S05,S06,S07,S08,S09], val_subj=S02, test_subj=S01 held
out) -- chosen over e.g. the S03 fold's higher raw Test_ROC_AUC (0.8982)
specifically so replaying S01 in `stream_infer.py`/`latency_harness.py` is
a genuine HELD-OUT sanity check against the offline
episode_recall=0.7826/mean_lead_time_s=0.6797 for S01, not an optimistic
replay through a model that already saw S01 during training. This is a
LOSO-fold artifact, not a final "deploy to any new patient" model -- that
would require a fresh full-data retrain (all 8 FOG subjects, no held-out
test), deliberately NOT done tonight (out of scope; tonight is pipeline
scaffolding, not model finalization).

**What was built and verified tonight (WSL/RTX4060, no physical sensor;
full numbers: results/reports/04_edge_deployment/edge_scaffold.txt):**
- `stream_infer.py`: ring-buffer streaming replay of a raw Daphnet file
  (data/dataset/S01R01.txt) sample-by-sample, CAUSAL by construction (the
  ring buffer is append-only, forward passes only ever see the current +
  past 63 samples), reusing `causal_median`/`causal_majority_vote` from
  `src/run_loso.py` UNCHANGED (not reinvented). 58 cue trigger events;
  13/18 onsets preceded by a cue within the 1.0s horizon (sanity
  recall=0.7222, mean lead=0.6899s) -- same ballpark as the offline
  reference, with documented, expected deltas (single-file replay vs the
  offline evaluation's two S01 files; continuous-through-freeze streaming
  vs the offline evaluation's dropped freeze-overlapping windows, ADR-014).
- `ras_cue.py`: a `CueStrategy` interface (abstract base, `trigger(event) ->
  list[beep offsets]`) with `FixedTempoCueStrategy` implemented
  (tempo_hz=1.8 placeholder, NOT tuned) and `AdaptivePhaseShiftCueStrategy`
  a documented `NotImplementedError` stub for later -- slots in without
  touching `RASCueEngine` or `stream_infer.py`. A real integration bug
  (event timestamps computed as `sample_index/fs_hz` against a
  WINDOW-indexed, not raw-sample-indexed, cue stream -- silently off by
  32x) was found and fixed during this session's own smoke testing, not
  left for tomorrow: `RASCueEngine.process_stream()` now accepts an
  explicit `sample_indices` array.
- `latency_harness.py`: measures per-step forward compute time (real,
  `time.perf_counter` + `torch.cuda.synchronize()` on GPU) and reports a
  FIXED STRUCTURAL constant (step_size/fs_hz = 500ms) for the post-proc
  buffering wait rather than fabricating a "measured" number for something
  a memory-speed replay cannot honestly time (a live IMU delivering samples
  at fs_hz is the only way to genuinely measure that wait -- Jetson,
  tomorrow, same harness). WSL results: forward mean=1.56ms (cuda) /
  1.20ms (cpu) -- CPU is actually faster here, expected for a model this
  small at batch_size=1 where GPU kernel-launch overhead dominates; whether
  Jetson's TensorRT engine changes that balance is exactly what tomorrow
  answers. Every print is explicitly labeled "WSL-BASELINE, NOT JETSON".
- `export_onnx.py`: written and complete (SigmoidWrapper bakes the sigmoid
  into the graph, opset=17 for TensorRT 8.5+/JetPack 5.x-6.x compatibility,
  fixed input shape [1,3,64], validates against 100 real S01 windows via
  onnxruntime asserting max abs diff < 1e-4) but BLOCKED: fog_env_wsl is
  missing the `onnx` package outright (confirmed via an isolated
  `torch.onnx.export` test before writing the script -- this blocks the
  export call itself, not just validation) and `onnxruntime`. Neither
  installed silently, per instructions -- exact fix reported:
  `pip install onnx onnxruntime`, then rerun.

**Status.** 3 of 4 files fully smoke-tested and passing on WSL; 1 file
(export_onnx.py) genuinely blocked on a missing dependency, not attempted
or faked -- clearly reported rather than silently skipped or worked around.
No changes to `src/` or `data/`. Jetson TODO (tomorrow): install onnx/
onnxruntime, rerun export_onnx.py for the real max-diff number, ONNX ->
TensorRT conversion, real IMU wiring, rerun latency_harness.py on-device
for the real Jetson latency figures.

## ADR-017 — Decouple deployment inference cadence (infer_step) from the training stride (STEP_SIZE) · RESOLVED (span-constant post-proc + refractory)
**Decision.** `edge/stream_infer.py` and `edge/latency_harness.py` gained an
`infer_step` parameter (default 8) that controls how often a forward pass
fires during DEPLOYED streaming inference, entirely independent of
`STEP_SIZE=32` -- the stride `src/fog_preprocessing.py`/`src/run_loso.py`
use to carve TRAINING windows out of a fixed, already-collected dataset.
The 64-sample causal window, the model architecture, and the trained
weights are all unchanged; `src/` and `data/` are untouched. Only how often
the already-trained model is allowed to look at a fresh window changes.

**Why these were ever the same number, and why they shouldn't be.**
`STEP_SIZE=32` exists to control window OVERLAP/density in a finite,
offline training dataset -- a training-time tradeoff between more training
examples (small stride) and less redundant/correlated windows (large
stride), with no relationship whatsoever to how fast a deployed model
*could* or *should* re-evaluate a live stream. `edge/stream_infer.py`
initially inherited `STEP_SIZE` as its forward-pass cadence purely because
it was the only stride value available, not because it was the right one
for deployment -- conflating "how the training set was windowed" with "how
often the deployed model should make a new decision" was an unexamined
default, not a deliberate choice. There is no dependency forcing them to
match: the model's receptive field and input shape (64 samples) are fixed
regardless of `infer_step`.

**Forward latency and decision interval are two DIFFERENT KINDS of number,
reported separately, never merged.** `edge/latency_harness.py`'s original
`end_to_end_ms = forward_ms + postproc_wait_ms` framing (edge_scaffold.txt,
part 1) obscured this: forward_ms is a MEASURED, hardware-dependent number
(what Jetson/TensorRT can actually change) while the buffering wait is a
STRUCTURAL, config-dependent number (`infer_step/fs_hz`, a design choice,
not a hardware limit). Summing them into one figure makes it look like a
single latency budget to optimize, when in reality one term is compute
you'd tune via quantization/engine choice and the other is a cadence you'd
tune via `infer_step` -- entirely different levers. `latency_harness.py`
now reports `forward_ms` (measured) and `decision_interval_ms` (structural)
as two separate, clearly labeled numbers for every `infer_step` in a sweep,
and explicitly does not offer a pre-summed "end_to_end" figure.

**Empirical result of decoupling (S01R01.txt replay, 18 onsets,
results/reports/04_edge_deployment/edge_scaffold_2.txt sections B/C):** sweeping
infer_step in {32, 16, 8} (decision_interval 500/250/125ms) shows forward
compute time is EMPIRICALLY FLAT (~1.2-1.7ms mean, both cpu and cuda,
independent of infer_step -- confirms the "same cost per call, different
call frequency" model) while streaming sanity recall improves monotonically
(0.7222 -> 0.8333 -> 1.0000) and mean lead time improves substantially
(0.6899s -> 0.8760s -> 0.8490s, non-monotonic between the last two but far
above the infer_step=32 baseline).

**The cost that is NOT free, and is NOT silently absorbed into "infer_step=8
wins":** causal_median(k=5) and causal_majority_vote(w=7) count DECISION
steps, not raw samples -- their real-time smoothing/debounce span is
`k*decision_interval_ms` / `w*decision_interval_ms`, which SHRINKS as
infer_step shrinks (combined worst-case span 5500ms -> 2750ms -> 1375ms).
The observed cue-trigger count rises sharply (58 -> 82 -> 226) as a direct
consequence -- some of that is genuinely "catching more true onsets"
(recall climbing to 18/18), but some of it is the same k=5/w=7 debounce
window now covering much less real time, making the binary cue stream
structurally chattier. k/w were never re-tuned for a shorter
decision_interval; they were originally chosen (ADR-004) around
STEP_SIZE=32's smoothing span. **Open, not resolved:** before committing to
a final infer_step for a live demo, either re-tune k/w so the real-time
smoothing span stays roughly constant across infer_step settings, or add an
explicit refractory/debounce period in `edge/ras_cue.py`. Flagged in
PROJECT_STATUS.md §4 item 12 rather than silently defaulting to
infer_step=8 as if it were a strictly free improvement.

**Status (original).** `infer_step` sweep (32/16/8) smoke-tested on WSL, all
three settings run end-to-end without error, causal post-proc confirmed
unchanged (median/majority_vote functions reused from `src/run_loso.py`,
same as ADR-016). No `src/` or `data/` changes. Full numbers:
results/reports/04_edge_deployment/edge_scaffold_2.txt.

**RESOLUTION (2026-07-04): span-constant post-proc + refractory debounce,
results/reports/04_edge_deployment/edge_scaffold_3.txt.** The open tradeoff above is now
resolved. `edge/stream_infer.py` gained `derive_kw(infer_step,
median_span_ms=800, majority_span_ms=1000)`:
```
k = max(1, round(median_span_ms   / decision_interval_ms))
w = max(1, round(majority_span_ms / decision_interval_ms))
```
so `causal_median(k)`/`causal_majority_vote(w)`'s REAL-TIME span stays
close to the target spans across `infer_step` instead of collapsing
(explicit `--k`/`--w` still override the derivation if needed). `edge/
ras_cue.py`'s `RASCueEngine` gained `refractory_ms` (default 2000ms):
after a cue event fires, rising edges within `refractory_ms` are
suppressed (counted, not re-emitted) -- reused unchanged by
`stream_infer.py` to report a WITH-refractory trigger count alongside the
raw count.

**Result: the trigger-count explosion is confirmed fixed** -- raw trigger
count now ranges 122-134 across infer_step in {32,16,8} (a 1.10x spread)
versus the pre-fix 58-226 (a 3.90x spread); combined worst-case post-proc
span is 1500/1500/1625ms (1.08x spread) versus the pre-fix 5500/2750/
1375ms (4x spread).

**A more important, non-obvious finding: most of the ORIGINAL "smaller
infer_step improves recall" trend (ADR-017's first status) was a smoothing-
span artifact, not a genuine cadence effect.** Once span is held
approximately constant, sanity recall does NOT keep improving with smaller
infer_step -- it is already at ceiling (18/18) at infer_step=32, stays at
ceiling at infer_step=16, and dips slightly to 17/18 at infer_step=8 (one
onset, plausibly sampling noise given only 18 total onsets, not a clear
trend). The likely explanation: fixed k=5/w=7 at infer_step=32 produced a
5500ms combined span, long enough to plausibly OVER-SMOOTH and dilute real
onset signal; shrinking infer_step under that FIXED k/w regime incidentally
shrank the span back toward a more reasonable range, and that de-dilution
-- not a direct cadence benefit -- was likely doing most of the work in the
original sweep's recall improvement. Mean lead time, in contrast, DOES show
a genuine, if modest, cadence benefit even with span held constant
(0.7240s -> 0.8351s -> 0.8667s as infer_step shrinks 32->16->8) -- finer
cadence lets the system register a threshold-crossing closer to when it
actually happens, a real effect distinct from the (now-resolved) smoothing
story.

**Refractory effect:** modest, real cleanup (~4-10% of raw triggers
suppressed at refractory_ms=2000 across the sweep), not a fix for trigger
VOLUME. **Flagged, not silently resolved:** with only 18 true onsets but
112-134 triggers even after refractory, most triggers are evidently NOT
clustered around the known onsets (a 2000ms refractory would suppress
tight bursts effectively if that were the pattern) -- they appear scattered
through the replay, consistent with this task's already-known low
window-level PR-AUC (baseline_prediction.txt: 0.093+/-0.058). This is a
PRECISION problem in the underlying probability stream, not something
cadence/span/refractory tuning alone can fix -- a separate, still-open
issue (PROJECT_STATUS.md §4 item 11) from the cadence/span question this
ADR resolves.

**Recommendation for the Jetson demo (for approval, not auto-committed as a
new code default anywhere):** infer_step=16, k=3, w=4 (derived,
span-constant), refractory_ms=2000. Matches infer_step=32's ceiling recall
(18/18) while giving a meaningfully better mean lead time (0.8351s vs
0.7240s, ~11% of the 1.00s horizon); infer_step=8 does not clearly
outperform it (slightly worse recall, a small further lead-time gain, ~2x
the forward passes, the highest raw trigger count of the three). Forward
compute cost is negligible on WSL/RTX4060 regardless of setting (~1.2-1.7ms/
pass, edge_scaffold_2.txt sec C) -- Jetson TODO: confirm this holds
on-device before finalizing.

**Status.** Span-constant derivation + refractory debounce implemented and
smoke-tested on WSL across the full infer_step sweep; no `src/` or `data/`
changes (edge/ only: `stream_infer.py`, `ras_cue.py`). Full numbers:
results/reports/04_edge_deployment/edge_scaffold_3.txt.

## ADR-018 — Optuna DETECTION study, scaled up: PR-AUC objective, S08 in tuning folds, 40 trials · LAUNCHED (overnight, in progress)
**Decision.** Scale up the ADR-013 Optuna smoke into an overnight,
unattended full study, with two substantive changes from the smoke config
(not just a bigger n_trials):

1. **Objective = mean val PR-AUC, not ROC-AUC.** New
   `inner_val_score_prauc()` in `src/tune_optuna.py` (same causal-median
   (k=5) smoothing as `run_loso.inner_val_score()`, `average_precision_score`
   instead of ROC-AUC). Rationale: this project's real weakness is
   precision under heavy class imbalance (window-level Test_PR_AUC ~0.09 on
   the PREDICTION task, and PR-AUC is consistently the DETECTION baseline's
   least-headroom metric too, ADR-007/S6) -- ROC-AUC is comparatively
   insensitive to that weakness because it does not reweight by the
   positive-class base rate the way PR-AUC does. Optimizing val PR-AUC
   pushes the search toward hyperparameters that improve probability-stream
   QUALITY under imbalance, not just rank-ordering. Does NOT touch
   `run_loso.py`'s own `inner_val_score()` or its ROC-AUC-based val_subj
   selection rule for the production DETECTION pipeline (ADR-003) -- that
   stays exactly as validated. `make_objective()` gained a `score_fn`
   parameter (default preserves the original ROC-AUC behavior for any old
   caller); only the scoring FUNCTION changes, the median-SELECTION RULE
   itself (rank inner candidates, pick the median) is identical either way.
2. **tuning_folds = [S01, S03, S08], not `fog_subjects[:3]` (=[S01,S02,S03]
   by alphabetical default).** S08 -- the akinetic-phenotype outlier,
   below-chance under DETECTION (ADR-012) -- is deliberately swapped in for
   S02 so the search is scored against a phenotype it has historically
   failed on, not just easy/average subjects. A hyperparameter set that
   only looks good on [S01,S02,S03] risks being tuned to the "easy"
   majority phenotype; including S08 pushes toward params that generalize
   across phenotypes.

**Scale:** 40 trials (up from the smoke's 2), TPE sampler (seed=42),
MedianPruner(n_startup_trials=8, n_warmup_steps=1) (up from 5, to give the
pruner a larger, more reliable sample before it starts aborting trials),
epoch_cap=25 (up from the smoke's 5 -- enough to let patience=5 early
stopping actually have room to trigger, unlike the 5-epoch smoke where it
structurally could not). Search space UNCHANGED from ADR-013 (same 4
RF<=64 arch combos, same dropout/focal_alpha/focal_gamma/lr/weight_decay/
batch_size ranges). Per-call unique checkpoint path (from the earlier
checkpoint-collision fix, ADR-013) inherited unchanged -- safe under
concurrent training jobs.

**Smoke gate (required before launch, per instructions -- run interactively
and verified BEFORE deciding to launch, not assumed):** 2 trials x 1 outer
fold (S01) x 5-epoch cap, objective=val PR-AUC. All 4 required checks
PASSED: RF guard rejects the known-bad combo (K5/[1,4,8], RF=105) and
accepts all 4 valid combos; objective demonstrably reads PR-AUC (trial
values 0.6201/0.5145, in PR-AUC's 0-1 imbalance-sensitive range, distinct
from the ROC-AUC objective's typical range); the VAL-ONLY leakage
assertion fired on every one of the 14 models trained (2 trials x 1 fold x
7 inner candidates) with ZERO AssertionErrors, both trials reaching
COMPLETE state; trials persisted to `study_smoke_prauc.db` +
`trials_log_smoke_prauc.csv`. Full detail: results/reports/02_optuna_tuning/optuna_launch.txt.

**Wall-clock projection (from the smoke's observed per-epoch cost):** 40
trials x 3 folds x 7 inner candidates x 25 epochs = ~28,917s (~8.03hr)
CONSERVATIVE UPPER BOUND -- assumes no early-stopping or pruning benefit.
The 5-epoch smoke could not exercise patience=5 early stopping at all (no
split ran long enough to hit 5 non-improving epochs), so this upper bound
does not yet reflect the savings the real 25-epoch-cap run should get from
early stopping AND from MedianPruner aborting bad trials after fold 1 of 3
once >=8 trials have completed. Real wall-clock is expected to be
meaningfully below 8.03hr. Fits comfortably in an overnight window either
way.

**Launch.** Detached in tmux session "optuna":
```
tmux new-session -d -s optuna "source fog_env_wsl/bin/activate && \
  export PYTHONUTF8=1 && export PYTHONUNBUFFERED=1 && \
  python -u src/tune_optuna.py --full > results/reports/06_run_logs/optuna_full.log 2>&1"
```
(Adjusted from the suggested `python -u -m src.tune_optuna --full` to
`python -u src/tune_optuna.py --full` -- this codebase's scripts have
always used flat `sys.path.insert(0, 'src')` imports inside the script
itself, not `-m` package-style invocation; `tune_optuna.py`'s own
`ROOT_DIR` resolution via `Path(__file__)` is unaffected by working
directory either way.) New `--full` CLI flag added to `src/tune_optuna.py`
with this task's exact config as named module-level constants
(`PRAUC_TUNING_FOLDS`, `PRAUC_N_TRIALS`, `PRAUC_EPOCH_CAP`,
`PRAUC_N_STARTUP_TRIALS`) so this ADR and PROJECT_STATUS.md can cite the
same values the launch actually used. Bare `python src/tune_optuna.py`
(no flag) still runs only the original ADR-013 ROC-AUC smoke, unchanged --
reproducibility of that earlier result is not disturbed by this change.

**Status at launch:** confirmed alive -- tmux session running, process
active on cuda, log header confirms the exact launched config (40 trials,
epoch_cap=25, objective=val_prauc, n_startup_trials=8,
tuning_folds=[S01,S03,S08]), first trial's first inner candidate observed
mid-training (epoch 7-8/25) with no errors. `results/optuna/
study_full_prauc.db` (sqlite) created and being written to incrementally
(crash-safe). Per-trial CSV: `results/optuna/trials_log_full_prauc.csv`
(appears once trial 0's first fold's 7 inner candidates complete). NOT
done tonight, deferred as instructed: the final 50-epoch run with best
params (`run_finalize()`, already implemented, unchanged) and any
precision-aware threshold work. No `data/` or `ROOT_DIR`/logging changes;
`src/` change limited to `src/tune_optuna.py`. Full launch report:
results/reports/02_optuna_tuning/optuna_launch.txt.

## ADR-019 — Edge platform choice: Yahboom Orin NX SUPER + Tailscale for remote access · ACCEPTED
**Decision.** Use the on-hand Yahboom Jetson Orin NX SUPER Developer Kit
(476GB SPCC NVMe SSD, expanded root from the stock 27GB to 467GB usable;
19V DC power mandatory; DP video out) as the edge deployment target, and
Tailscale as the primary remote-access path (SSH to the Jetson's tailnet
address 100.97.43.118, tailnet candy070405@, sudo user "stu"), with wired
ethernet (140.116.132.43, NCKU campus network) as a same-network fallback.
OS: L4T R36.4.7 (JetPack 6.2), Ubuntu 22.04, CUDA 12.6, Python 3.10.12.
Workspace: `~/fog_edge/{artifacts,scripts,logs,models}` on-device, kept
separate from the device's previous user's `~/jps/` (not touched -- unrelated
prior work). Full detail: PROJECT_STATUS.md §9.

**Why this hardware.** It is already on hand -- no procurement delay
against the end-of-August deadline. JetPack 6.x ships TensorRT 10.x, which
supports ONNX opset 17 -- exactly the opset `edge/export_onnx.py` already
exports at (ADR-016), so the existing `edge/artifacts/fog_tcn.onnx`
artifact should convert without needing an opset downgrade/re-export step.

**Why Tailscale over campus VPN.** The Jetson physically stays at NCKU;
development continues remotely from Kaohsiung. Tailscale gives a stable,
zero-config SSH path that works identically regardless of which network
either end is on (no campus-VPN client/credential dependency, no NAT/
port-forwarding setup on either side) -- the wired NCKU-campus IP
(140.116.132.43) only works when both ends happen to be on that network,
so it is kept only as a fallback, not the primary path.

**Known issue, not a blocker.** Tailscale logs an iptables-legacy connmark
warning on this device; a system restart is pending to clear it. SSH and
basic connectivity are confirmed working despite the warning -- flagged
here so a future session doesn't mistake it for a new problem if it
resurfaces after the pending restart.

**Status.** Platform setup done; this ADR + PROJECT_STATUS.md §9
consolidate that state before on-device work begins (ONNX->TensorRT
conversion, real IMU wiring, on-device latency measurement per
ADR-016/017). No training run, no `data/` or `src/` changes this session --
read-only documentation consolidation only.

## ADR-020 — Adopt Optuna trial 26 (not trial 33) for the tuned DETECTION baseline candidate · ACCEPTED (candidate, not yet promoted)
**Decision.** Run the full 8-fold inner-LOSO DETECTION protocol
(RUN_20260705_151423) using Optuna trial 26's hyperparameters -- the best
CLEAN trial from the ADR-018 overnight study -- NOT trial 33, which had the
nominally higher tuning-time score. Both trial 26 and the resulting full
run are treated as a CANDIDATE alongside the existing official baseline
(RUN_20260702_202440), not a replacement -- see PROJECT_STATUS.md §3.

**Why trial 26, not trial 33 (full audit: results/reports/02_optuna_tuning/optuna_summary.txt).**
Trial 33's nominal top val PR-AUC (0.6260) came from a NaN-contaminated
run: its S08-fold selected (median-rule) candidate diverged to NaN loss at
epoch 16 during the Optuna study and was rescued by the checkpoint/early-
stopping mechanism (the reported number is not garbage -- the rescue
mechanism reloads the last valid pre-divergence checkpoint before
scoring -- but it reflects a run that became unstable, not one that
trained cleanly for the full epoch cap). Trial 26 is the best trial with
ZERO divergence across all 21 candidate trainings in its Optuna
evaluation, at val PR-AUC=0.6173 -- a gap of only 0.0087 from trial 33,
smaller than the top-5 trials' own spread (0.0113) and well within the
field's overall noise (std=0.0408). Given that gap is not clearly
distinguishable from noise once NaN-rescue risk is discounted, the CLEAN
trial is the defensible choice for a hyperparameter set going into a
production-candidate run, even though it was not the literal top of the
tuning leaderboard.

**Hyperparameters (trial 26):** kernel_size=5, dilations=[1,2,4],
num_channels=[32,64,128] (architecture UNCHANGED from the baseline,
RF=57), dropout=0.188296, focal_alpha=0.747900, focal_gamma=2.020007,
lr=0.000780, weight_decay=0.000699, batch_size=256. n_epochs=50, seed=42 --
same as the baseline in every respect except these 6 hyperparameters.

**Implementation:** no `src/` logic duplicated. `run_loso.py`'s existing
`main()` already accepted every one of these 7 parameters (added for
ADR-013's Optuna work) -- a new `scripts/launch_final_detection_trial26.sh`
(mirroring `scripts/launch_full_inner_loso.sh` exactly) simply calls
`main()` with trial 26's values baked in. No `--final`/`--config` CLI flag
was added to `run_loso.py` since the existing shell-launch-script
convention already covers this without any new code path.

**Result (RUN_20260705_151423, full 8-fold, 2795.1s wall-clock, 0 NaN
epochs, 0 leakage/error signatures across all 64 models -- full table:
results/reports/01_baselines/baseline_tuned.txt):**

| Metric | Baseline (202440) | Tuned (trial26) | Delta | Std |
|---|---|---|---|---|
| Test_ROC_AUC | 0.7743 +/- 0.1261 | 0.8660 +/- 0.0734 | +0.0917 | SHRANK |
| Test_PR_AUC | 0.4211 +/- 0.1081 | 0.5531 +/- 0.1573 | +0.1320 | GREW |
| Test_F1 | 0.3664 +/- 0.1439 | 0.3865 +/- 0.2202 | +0.0201 | GREW |

ROC-AUC's combination of higher mean AND lower variance is the same
signature that validated the ADR-003 inner-LOSO fix -- the most
trustworthy result here. PR-AUC's larger absolute gain came with more
cross-fold variance (real but uneven improvement). F1's small gain and
sharply grown std reflect a genuine, unresolved issue: 3 of 8 folds (S06,
S07, S08) have low F1 despite good-to-great ROC-AUC under these
hyperparameters -- the val-frozen (Youden's J) threshold is evidently not
transferring to test as reliably as it did under the original baseline's
hyperparameters. Flagged as PROJECT_STATUS.md §4 item 13, NOT resolved by
this ADR.

**Notable, unplanned finding:** S08 (the akinetic-phenotype outlier,
below-chance under the original baseline at Test_ROC_AUC=0.4908, ADR-012)
rises to 0.7321 under trial 26's hyperparameters -- the single largest
per-fold change in the comparison (+0.2413), moving it from below-chance
to comfortably above 0.70. This was not a specific target of the Optuna
objective (S08 was one of 3 tuning folds, but the objective was the MEAN
across folds). Its F1 remains low (0.0909), consistent with the ROC-AUC-
improved/F1-inconsistent pattern above -- ranking quality improved
dramatically, threshold calibration for this subject did not.

**Learning-rate flag (reaffirmed):** trial26's lr=0.000780 is below the
1e-3 threshold ADR-018 identified as a near-perfect predictor of NaN
divergence in this search space -- consistent with this run producing 0
NaN epochs across all 64 models, further supporting that flag for any
future study's search-space design.

**Status.** Candidate produced and documented; NOT promoted to the
official baseline (PROJECT_STATUS.md §3 keeps both runs side by side).
Promotion, and resolution of the F1/threshold-transfer issue, are open
follow-ups for a future session.

## ADR-021 — FP32 TensorRT engine adopted as the Jetson deployment target, not FP16 · ACCEPTED
**Decision.** `fog_tcn_fp32.engine` is the deployment target on Jetson
Orin NX SUPER; `fog_tcn_fp16.engine` was built (Task 3) but is NOT adopted.

**Rationale:**
- **No speedup from FP16 on this model.** trtexec's own benchmark (Task 3,
  `results/reports/04_edge_deployment/jetson_trt_build.txt`) found FP32 and
  FP16 GPU Compute times statistically indistinguishable for this TCN
  (small channel counts, kernel_size<=5, window=64, batch=1) -- the model
  is launch-bound, not compute-bound, at this size, so FP16's usual
  throughput advantage never materializes. Confirmed independently by this
  session's own Python-level latency harness (Task 5,
  `results/reports/04_edge_deployment/jetson_latency.txt`), which only
  exercises the FP32 engine but reports forward-pass timings (median
  0.6292ms) consistent with trtexec's FP32 number scaled for the extra
  Python/ctypes call overhead -- nothing suggests FP16 would have done
  meaningfully better.
- **FP32 is numerically closer to the source ONNX.** Task 4's parity check
  (`results/reports/04_edge_deployment/jetson_verify_engine.txt`) found FP32
  max abs diff = 3.51e-05 vs ONNX (well under the 1e-4 threshold), while
  FP16's max abs diff = 2.08e-04 (also passes its own looser 1e-2
  threshold, but is ~6x larger in absolute terms than FP32's error). Given
  FP16 buys nothing on speed, there is no reason to accept its larger
  numerical deviation from the validated PyTorch->ONNX chain.
- **Engine size difference is negligible.** FP32 = 922,228 bytes vs FP16 =
  653,404 bytes (a mixed-precision engine, not a clean 50% FP16 reduction,
  per Task 3's finding that TensorRT's own autotuner picked FP32 kernels
  for much of this tiny network anyway). Neither size is a storage/memory
  concern on an 8GB+ Jetson.

**Status.** FP16 engine is retained on disk (built, not deleted) as a
reference artifact only; all subsequent Jetson work (Task 5's latency
harness, and any future demo/deployment) uses FP32 exclusively.

## ADR-022 — Post-processing kept in pandas despite being ~2x slower than the forward pass on Jetson · ACCEPTED (deferred optimization)
**Finding.** Task 5's on-device harness
(`results/reports/04_edge_deployment/jetson_latency.txt`) measured
`causal_median`/`causal_majority_vote` (pandas `Series.rolling()` over a
tiny k=3/w=4 buffer) at ~1.19ms median per decision step -- roughly 1.9x
the TensorRT FP32 forward pass's own 0.63ms median. Pandas per-call object-
creation overhead, not the rolling computation itself, is the likely
driver (the buffers involved are only 3-4 elements long).

**Decision.** Keep the existing pandas-based `causal_median`/
`causal_majority_vote` (src/run_loso.py, reused unchanged by the edge/
pipeline, ADR-004) as-is. Do NOT rewrite them (e.g. as a hand-rolled NumPy
deque-based incremental update) in this session.

**Rationale.** End-to-end compute (forward + post-proc combined) is still
only ~1.82ms median against a 250ms structural decision interval at the
approved infer_step=16 config -- compute occupies under 1% of the
available time budget either way. Optimizing a component that is already
2 orders of magnitude under its budget would be solving a problem that
does not exist yet. Revisit only if a future, meaningfully smaller/slower
target platform (a lower-power Jetson variant, or a non-GPU MCU-class
target) makes the current comfortable margin disappear -- not before.

---
### Open blockers feeding these ADRs
- (none — ADR-003/004/005 resolved 2026-06-15, see entries above)
