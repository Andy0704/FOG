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
  — extra sensors barely help. Open daAZ\
  ta + software. → justifies our single-ankle
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

## WOS search — FoG methods, RAS, cross-dataset (added 2026-07-13)

### FoG Detection Methods & Architectures

- **Djuric-Jovicic et al. 2014** — Automatic Identification and Classification of
  Freezing of Gait Episodes in Parkinson's Disease Patients. *IEEE Trans Neural Syst
  Rehabil Eng.* DOI: 10.1109/TNSRE.2013.2287241 · cites 74 (WOS).
  [METHODOLOGY] Rule-based stride classifier on shank IMU (100 Hz); distinguishes
  trembling-FoG (FOG+) vs akinetic/motor-block FoG (FOG−), both 100% identification
  rate. Key for ADR-012 / case_study_S08: akinetic FoG has weaker 3–8 Hz spectral
  energy than trembling FoG — explains why frequency-domain and DL models trained on
  trembling-dominant data underperform on S08's quiet akinetic phenotype.

- **Hwang et al. 2025** — Efficiently Identifying Non-FoG, Pre-FoG, Pre-FoG Transition,
  and FoG in Parkinson's Disease Patients Using Window Acceleration and Spline Function
  Features. *IEEE Trans Neural Syst Rehabil Eng.* DOI: 10.1109/TNSRE.2025.3618400
  · cites 0 (WOS, just published).
  [METHOD][METHODOLOGY] Uses Daphnet dataset (64 Hz, ankle+thigh+trunk, 10 subjects);
  dynamic label assignment for pre-FoG stages; window overlap sensitivity analysis:
  overlap < 25% → low overfitting risk; Pre-FoG sensitivity 96% / specificity 88%.
  ⚠️ EVALUATION PROTOCOL: subject-dependent 70/30 random window split (NOT LOSO) —
  authors explicitly acknowledge this as a limitation and note LOSO requires more FoG
  events. The 96% figure CANNOT be compared directly to our LOSO AUROC numbers. Cite
  with qualifier "subject-dependent setting". Directly relevant to ADR-014 (prediction
  framing) and window overlap choice.

- **Yang et al. 2024** — Freezing of gait assessment with inertial measurement units
  and deep learning: a multi-center study. *J NeuroEngineering Rehabil.*
  DOI: 10.1186/s12984-024-01320-1 · cites 0 (WOS, recent).
  [METHOD] TCN (Initial block, k=3, dilations=[1,3,9,27,81], RF=243 samples=3.8 s @
  64 Hz) + MS-TCN refinement on 5-IMU (pelvis, bilateral tibia+talus) at 64 Hz; 4 s
  window; LOSO; primary metrics: Sample-F1, Segment-F1@50, %TF ICC=0.92, #FOG ICC=0.95.
  ⚠️ NON-CAUSAL: symmetric padding 121 samples each side → requires 1.89 s of future
  data; designed as offline annotation tool, not real-time. Subject S8 = 0 FoG events
  (exact S04/S10 analog; handled differently from our ADR-002). Systematic overestimation
  bias (mean bias −4.06 %TF): festination/shuffling misclassified as FoG — same false-
  alarm source our refractory period (2000 ms) targets. Direct comparator: our system is
  causal, single-ankle, 1 s window, 1.82 ms Jetson latency.

- **Sigcha et al. 2024** — Deep learning algorithms for detecting freezing of gait in
  Parkinson's disease: A cross-dataset study. *Expert Systems With Applications.*
  DOI: 10.1016/j.eswa.2024.124522 · cites 0 (WOS, recent).
  [METHOD][METHODOLOGY] CNN-MLP / Wide-CNN on Rempark+Daphnet+O'Day unified to 32 Hz
  via FIR Kaiser (β=5) anti-aliasing LPF with offline delay compensation; Daphnet
  single-dataset AUROC: CNN-MLP 0.852 / Wide-CNN 0.844; cross-dataset (test=Daphnet)
  Wide-CNN 0.839 (stable); cross-dataset (test=O'Day) Wide-CNN 0.654 (collapsed).
  Domain shift root cause confirmed as subject heterogeneity (ANOVA p<0.001 feature
  distribution differences), NOT sensor placement or sampling-rate differences.
  Fills Query-2 null result: only FoG paper found that explicitly documents cross-dataset
  sampling-rate alignment strategy. FIR group delay not stated in ms (offline compensation
  only — real-time deployment must independently budget group delay). Supports new ADR
  for anti-aliasing when hardware rate > 64 Hz.

### Clinical Motivation — RAS Cueing

- **Ghai et al. 2018** — Effect of rhythmic auditory cueing on parkinsonian gait:
  A systematic review and meta-analysis. *Scientific Reports.*
  DOI: 10.1038/s41598-017-16232-5 · cites 0 (WOS; >300 Google Scholar).
  [REVIEW] Meta-analysis of RCTs; RAS improves gait velocity (large effect), cadence,
  and stride length in PD; strongest evidence for sustained over-ground walking. Use in
  Introduction to justify closed-loop RAS cueing as the intervention target.

- **Nieuwboer et al. 2007** — Cueing training in the home improves gait-related mobility
  in Parkinson's disease: the RESCUE-trial. *J Neurol Neurosurg Psychiatry.*
  DOI: 10.1136/jnnp.2006.097923 · cites 0 (WOS; landmark RCT).
  [METHODOLOGY] Multicenter RCT (n=153); 3-week home RAS training significantly improves
  FoG frequency and gait speed; gold-standard clinical evidence that RAS reduces FoG.
  Use alongside Ghai 2018 in Introduction clinical-motivation paragraph.

### Wearable Monitoring & Free-Living Deployment

- **Del Din et al. 2016** — Free-living monitoring of Parkinson's disease: Lessons from
  the field. *Movement Disorders.* DOI: 10.1002/mds.26718 · cites 317 (WOS).
  [REVIEW] Authoritative review (top PD clinical journal): no fully validated free-living
  FoG detection system exists; standardized definitions, protocols, and algorithm
  validation flagged as key gaps. High-citation anchor for Introduction motivation
  paragraph: "why real-world wearable + edge deployment remains unsolved."

- **Costa et al. 2026** — Biomechanical characteristics before, during, and after
  freezing of gait episodes in individuals with Parkinson's disease. *Gait & Posture.*
  DOI: 10.1016/j.gaitpost.2026.110171 · cites 0 (WOS, in press).
  [METHODOLOGY] Single lumbar IMU (Axivity AX3, 100 Hz), DeFoG dataset, 2 s windows;
  AP-axis FoG-ratio peaks at FoG; AP+V RMS already elevated in pre-FoG window.
  Supports physiological justification for our 1 s prediction horizon: pre-FoG
  biomechanical signatures are detectable before episode onset. Sensor site differs
  (lumbar vs. our ankle); AUC moderate (AP RMS 0.61, V RMS 0.63) vs. our DL approach.

## Edge deployment direct comparator (added 2026-09-12)

- **Borzì et al. 2025** — Edge-based freezing of gait recognition in
  Parkinson's disease. *Computers and Electrical Engineering*, 127, 110530.
  DOI: 10.1016/j.compeleceng.2025.110530 · cites 2–3 (Scopus, recent).
  [METHOD][DEPLOYMENT] Lightweight 1D-CNN (4,641 params) deployed on Nordic
  Thingy:53 MCU (nRF5340, Edge Impulse, int8 quantized): 1ms inference,
  6.3 KB RAM, 37.8 KB Flash. Single lower-back IMU, downsampled to 32 Hz.
  Window-level: AUROC 0.966, F1=92.2%. Episode-level: 22% predicted ~2s
  before onset, 47% detected with ~1s delay, 13.5% missed.
  Validation: 50/25/25 subject-split (NOT LOSO) on proprietary 62-subject
  dataset + external Rempark dataset.
  ⚠ KEY DISTINCTIONS vs. this project:
    (1) Subject-split validation, not nested LOSO — generalization claim weaker.
    (2) No causal architecture specified; Global Average Pooling used.
    (3) Detection task only — 22% early detections are incidental, not a
        designed label-shift prediction framework.
    (4) No closed-loop RAS cueing or on-body human test reported.
    (5) MCU-class edge (sensor node) vs. our GPU-class edge (gateway device);
        latency figures not directly comparable (int8/MCU vs. FP32/Jetson).
  Brief framing: "Borzì et al. (2025) demonstrate MCU-level edge deployment
  of FoG detection but use subject-split validation and lack causal design and
  prediction capability. Our work additionally provides nested LOSO validation,
  strictly causal inference, and closed-loop prediction verified on a human body."
