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

## FoG edge deployment — IMU, sampling rate, window length (added 2026-07-09)
- **Abbasi & Rezaee 2024** — Deep Learning–Based Prediction of Freezing of Gait in
  Parkinson's Disease With the Ensemble Channel Selection Approach. *Brain and
  Behavior.* DOI: 10.1002/brb3.70206 · cites 12.
  [METHOD] CBA-BiLSTM on ankle+leg+trunk sensors; ensemble channel selection +
  attention mapping reduces to 2 channels at 99.88% accuracy, enabling real-time
  monitoring. Alternative evidence (site *reduction via learning* vs. our fixed
  single-ankle) for ADR-009. Causal post-processing not stated in abstract.
- **Delgado-Terán et al. 2025** — Ankle Sensor-Based Detection of Freezing of Gait
  in Parkinson's Disease in Semi-Free Living Environments. *Sensors.*
  DOI: 10.3390/s25061895 · cites 10.
  [DATASET][METHODOLOGY] Single-ankle IMU + CNN, 24 PD participants; AUROC 0.9596
  (5Fold-CV) / 0.9275 (LOSO-CV) on walking+turning, dropping to 0.89/0.90 across
  all activities in semi-free-living conditions. Real-world stress-test of the
  single-ankle assumption (ADR-009) beyond lab-only O'Day 2022 evidence.
- **Gregorčič & Georgiev 2025** — The Usefulness of Wearable Sensors for Detecting
  Freezing of Gait in Parkinson's Disease: A Systematic Review. *Sensors.*
  DOI: 10.3390/s25165101 · cites 3.
  [REVIEW] 43 studies synthesized; accel+gyro most common combo (best config
  ~100% sens/spec); waist/ankles/shanks/feet most common sites; no standardized
  placement guideline; free-living validation flagged as the open gap. Directly
  updates the Huang et al. 2023 review above with 2025 placement-standardization
  status for ADR-009 framing.
- **Koltermann et al. 2024** — Gait-Guard: Turn-Aware Freezing of Gait Detection
  for Non-Intrusive Intervention Systems. *2024 IEEE/ACM CHASE.*
  DOI: 10.1109/chase60773.2024.00016 · cites 5.
  [METHOD][DEPLOYMENT] Closed-loop real-time FoG detection + intervention; 378.5 ms
  average intervention latency, 96.5% TP rate, subject-independent, 26 patients /
  1591 events. Concrete end-to-end latency budget target for our Jetson streaming
  path; causal by construction (closed-loop cueing requires past-only processing).
- **Kita et al. 2017** — Reliable and Robust Detection of Freezing of Gait Episodes
  With Wearable Electronic Devices. *IEEE Sensors Journal.*
  DOI: 10.1109/jsen.2017.2659780 · cites 32.
  [METHOD] Early real-time wearable MEMS-IMU FoG detector, >97% specificity,
  indoor/outdoor. Foundational real-time-constraint precedent predating
  window/RF-based deep nets; causal by design (real-time embedded operation).
- **Al-Adhaileh et al. 2025b** — Deep learning techniques for detecting freezing of
  gait episodes in Parkinson's disease using wearable sensors. *Frontiers in
  Physiology.* DOI: 10.3389/fphys.2025.1581699 · cites 11.
  [METHOD][DEPLOYMENT] CNN-BiLSTM+attention on tDCS-FOG/DeFOG/Daily-Living/Hantao
  multimodal sets; quantized + pruned for Raspberry Pi / Coral TPU, inference
  latency <350 ms. Direct Jetson-class edge-latency comparator for our deployment
  numbers. Distinct paper from the WaveNet Al-Adhaileh et al. 2025 entry above
  (different DOI/venue) — labeled "2025b" to disambiguate same-author-year. Causal
  post-processing not stated in abstract — flagged unconfirmed.
