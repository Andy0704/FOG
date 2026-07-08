# CLAUDE.md — FoG TCN Project Memory & Behavioral Contract

These rules apply to every task in this repo unless explicitly overridden in chat.
Bias: caution over speed on anything that touches data splits, metrics, or the
training pipeline. Use judgment on trivial edits (plots, comments, docstrings).

No role-play preamble on purpose. Empirically, "act as a senior engineer" /
"think carefully" instructions are noise and lower compliance. Rules below are
concrete and verifiable instead.

---

## 0. Project context (1 paragraph)
Real-time biomedical signal algorithm on a wearable IMU that PREDICTS Freezing
of Gait (FoG) in Parkinson's patients, intended as the controller of a
closed-loop neuro-rehab system (RAS cueing). Core model = Temporal Convolutional
Network (dilated causal convs) chosen over LSTM for low latency. Primary dataset
= Daphnet (64 Hz, 10 subjects, 8 with FoG). Target deployment = Jetson Orin NX
8 GB + single ankle IMU. Hard deadline: end of August (推甄 portfolio + conference-
grade write-up). See PROJECT_STATUS.md for live state, DECISIONS.md for the why
behind every methodological choice, EXPERIMENTS.md for the run log.

---

## 1. NON-NEGOTIABLE SCIENTIFIC RULES (a violation invalidates results)
- **S1 Subject-level isolation.** No window may leak across subjects between
  train / val / test. Cross-subject leakage is a critical failure, not a bug.
- **S2 Threshold on validation only.** The decision threshold is chosen on the
  VALIDATION subject(s), then frozen and applied to the TEST subject. NEVER pick
  the threshold on the test subject. "Best-F1-swept-on-test" may appear ONLY as a
  clearly labeled upper bound, never as the headline number. (Current baseline
  violates this — see DECISIONS ADR-003.)
- **S3 Task framing is explicit.** Every run states in code + logs whether it is
  DETECTION (label at t) or PREDICTION (label shifted +H into the future).
  Do not call detection "prediction".
- **S4 Causal post-processing for any deployable number.** Median filter (K) and
  rolling majority vote (W) must be PAST-ONLY (causal) for metrics claimed as
  real-time. Report total latency = model_forward + post-proc window wait, not
  just forward time.
- **S5 Receptive-field check matches architecture.** Before instantiating any TCN,
  assert RF <= window_size. RF = 1 + n_conv_per_block * sum((K-1) * dilation_l).
  Read n_conv_per_block from the model config; never hardcode the single-conv form.
- **S6 Metrics under imbalance.** Daphnet is ~81/19. Always report PR-AUC (AP) in
  addition to ROC-AUC, F1 at the frozen threshold, plus episode-level detection
  rate and detection latency.

## 2. Behavioral rules (adapted from the 12-rule CLAUDE.md, trimmed to what
##    actually prevents errors in THIS repo)
- **R1 Think before coding.** State assumptions explicitly. If a request is
  ambiguous, ask rather than guess. Stop when confused and name what's unclear.
- **R2 Minimal change, but research is allowed.** Solve the asked problem with the
  least code. No speculative production abstractions for one-off experiments —
  but throwaway experiment scripts and scaffolding are fine and expected.
- **R3 Surgical edits.** Touch only what the task needs. Do NOT "improve" adjacent
  preprocessing, the LOSO loop, or formatting while doing something else. The
  working pipeline is load-bearing.
- **R4 Verifiable success criteria.** Define success as a concrete check (a metric
  on a held-out split, an assert that passes, a shape that matches). Iterate to it.
- **R5 Deterministic logic stays in code.** Splits, RF checks, label mapping,
  metric computation, seeding — plain deterministic Python, never "vibes".
- **R6 Token/session budget is real.** When context runs low (you WILL hit session
  limits), STOP and write the current state into PROJECT_STATUS.md before it's
  lost. A fresh session must be able to resume from those .md files alone.
- **R7 Surface conflicts, don't average.** If old and new preprocessing/config
  coexist, pick one (the tested one), say why, flag the other. Never blend.
- **R8 Read before you write.** Before editing a file (esp. fog_preprocessing.py,
  trainer_tcn.py, run_tcn_v2.py), read its inputs/outputs and callers first.
- **R9 Tests verify INTENT (ML version).** A useful test here asserts: no subject
  overlap across splits; RF <= window; label mapping (Daphnet 0=invalid,
  1=no-FoG, 2=FoG) is correct; post-proc is causal. A test that can't fail when
  the split logic breaks is worthless.
- **R10 Checkpoint after each significant step.** After each step: summarize what
  was done, what's verified, what's left. Don't continue from a state you can't
  describe back.
- **R11 Match conventions.** Python, snake_case, existing module layout. If you
  think a convention is harmful, raise it in chat; don't silently fork it.
- **R12 Fail loud.** "Run finished" is wrong if a subject was silently NaN'd or
  skipped. "Metrics improved" is wrong if the threshold was picked on test.
  Default to surfacing uncertainty.
- **R13 (added 2026-07-05)**: When encountering an unknown filesystem state (ghost entries,
  stat/ls disagreement, unexpected orphans), STOP and report. Never invoke rm/rmdir/
  git rm to "clean up" a state you can't fully explain. This binds strictly for any
  path under data/, src/, results/, or any location whose contents pre-exist the
  current session. The loss of optuna_smoke_raw.log during the 2026-07-05 reports/
  reorg is the precedent.

## 3. Engineering standards
- PyTorch 2.x, scipy, torchmetrics, pandas, numpy. Training GPU = local RTX 4060.
- Reproducibility: seed torch / numpy / random; report final metrics as
  mean ± std across >= 3 seeds. Log the exact config of every run.
- `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
- `torch.load(..., weights_only=True)` always.
- `from torch.nn.utils.parametrizations import weight_norm` (not deprecated util).
- Logging: summary -> list[dict] -> pd.DataFrame; per-sequence -> list[DataFrame]
  -> pd.concat. No row-by-row append.
- Every experiment appends ONE row to EXPERIMENTS.md (run_id, config, key metrics,
  one-line takeaway).
- Preprocessing must keep toggle hooks ready for: downsample to 32 Hz,
  scaling-to-g, magnitude = sqrt(x^2+y^2+z^2) as a 4th channel.

## 4. Current focus (update me as it changes; mirror in PROJECT_STATUS.md)
1. Fix S2 (validation-chosen threshold) and re-report the post-processing gain
   honestly; this gates the credibility of the "+4% F1" claim.
2. Reframe detection -> prediction via label-shift, horizon = 1 s (64 samples).
3. Add cross-dataset external validation (DeFOG / O'Day) — keep Daphnet primary.
4. Build a task-agnostic streaming inference path for Jetson (works with either a
   detection or prediction model) before wiring RAS cueing.

## 5. Working mode
- Reply to the human in Traditional Chinese; keep all code, identifiers, and these
  .md files in English.
- Chat and CLI are separate workflows. When the human is in chat, propose; when in
  CLI/extension, execute against the repo following the rules above.
