# References — FoG / wearable IMU / TCN

Curated from nrp-literature MCP (Semantic Scholar + OpenAlex). Tags:
[DATASET] candidate/data, [METHOD] model/architecture, [METHODOLOGY] eval/protocol,
[REVIEW] background. Relevance lines are our own notes, not the abstract.

> CLI task: keep this lean. New papers append here via the `lit-review` skill,
> deduped by DOI. Full per-paper notes (if needed) go in docs/literature/<key>.md.

---

## Datasets & sensor placement
- **O'Day et al. 2022** — Assessing IMU locations for FoG detection and patient
  preference. *J NeuroEng Rehabil.* DOI: 10.1186/s12984-022-00992-x · cites 69.
  [DATASET][METHODOLOGY] Single-ankle IMU AUROC 0.80 vs 3-IMU (lumbar+ankles) 0.83
  — extra sensors barely help. Open data + software. → justifies our single-ankle
  plan (ADR-009) and is a candidate open dataset.
- **Mancini et al. 2021** — Measuring FoG during daily life, open-source wearable
  approach. *J NeuroEng Rehabil.* DOI: 10.1186/s12984-020-00774-3 · cites 184.
  [DATASET] Feet+lumbar, 40 (lab) + 48 (home) PD. Larger, daily-life. Candidate.
- **Bächlin et al. 2010** — Wearable assistant for PD with FoG (Daphnet origin).
  *IEEE TITB.* (PDF in data/doc/) [DATASET] Our current primary dataset; 10 PD,
  8 with FoG, 3 accel @64 Hz (ankle/thigh/trunk), ~81/19 imbalance.

## Models / methods (TCN-relevant)
- **Al-Adhaileh et al. 2025** — WaveNet framework for real-time FoG prediction.
  *J Disability Research.* DOI: 10.57197/jdr-2025-0722 · cites 0 (new).
  [METHOD] WaveNet = dilated causal convs (same family as our TCN); benchmarked on
  tDCS-FoG + DeFoG; ADASYN + jerk features for imbalance; low latency, lightweight.
  → closest published analogue to our architecture; primary Related-Work anchor.
- **Alsarraj et al. 2025** — Stacked LSTM-GRU PD auxiliary diagnosis on DeFOG.
  *JOWUA.* DOI: 10.58346/jowua.2025.i3.028 · cites 0 (new).
  [METHOD][DATASET] DeFOG, ~87% F1 / AUC 0.92, 8 ms/sample → real-time. Baseline to
  beat and a DeFOG ingestion reference.
- **Shalin et al. 2021** — FoG prediction & detection from plantar pressure (LSTM).
  *J NeuroEng Rehabil.* DOI: 10.1186/s12984-021-00958-5 · cites 96.
  [METHODOLOGY] Clean detection-vs-prediction split (pre-FoG window for prediction);
  leave-one-freezer-out CV; tests on non-freezers for specificity → mirrors our
  S04/S10 false-alarm suite (ADR-002) and the prediction reframe (ADR-006).
- **Bikias et al. 2021 (DeepFoG)** — Single wrist IMU, deep learning FoG detection.
  *Front Robot AI.* DOI: 10.3389/frobt.2021.537384 · cites 106.
  [METHOD] Single IMU (accel+gyro), 11 PD, LOSO 83/88 sens/spec; explicitly motivates
  RAS + vibration cueing → supports our single-IMU + closed-loop framing.
- **Ren et al. 2022** — FoG recognition from combined wearable sensors (RF).
  *BMC Neurol.* DOI: 10.1186/s12883-022-02732-z · cites 35.
  [METHOD] Sensor-config study; left-shank gyro+accel, 35 features → feature/placement
  reference for our 4th-channel (magnitude) and IMU choice.

## Generalization / fairness
- **Odonga et al. 2025 (Kwon lab)** — Bias & fairness in FoG detection, mitigation.
  *arXiv.* DOI: 10.48550/arxiv.2502.09626 · cites 0 (new).
  [METHODOLOGY] Multi-site transfer learning improves both fairness and F1; threshold
  optimization / adversarial debiasing fail. → evidence for cross-dataset eval (ADR-008b).

## Reviews / background
- **Huang et al. 2023** — Systematic review: wearables for FoG & fall detection.
  *Front Aging Neurosci.* DOI: 10.3389/fnagi.2023.1119956 · cites 45.
  [REVIEW] Thigh+ankle most common placement; accel+gyro most common; trend toward
  ML; calls for free-living validation. Good intro-section citation.
