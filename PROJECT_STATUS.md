# PROJECT_STATUS.md — Live State (the cross-session "save file")

> Any fresh Claude session should be able to resume from THIS file + DECISIONS.md
> + EXPERIMENTS.md alone. Update the "Last updated" line and the relevant section
> at the end of every working session (R6/R10 in CLAUDE.md).

**Last updated:** 2026-06-14 (initialized)
**Phase:** End of Phase 2 (Week 4–5), now pivoting toward prediction + edge deploy.
**Hard deadline:** end of August — 推甄 portfolio + conference-grade write-up.

---

## 1. Snapshot
- Core model: TCN (dilated causal convs), channels [32, 64, 128], window = 64
  samples (1 s @ 64 Hz), Binary Focal Loss, Adam, early stopping.
- Validation: subject-level LOSO over Daphnet.
- Post-processing: median filter (K=5) + rolling majority vote (W=7) on the
  probability output (reported as +4% mean Best-F1, −2% AUC — see caveat below).

## 2. Done (Weeks 1–3)
- Daphnet 64 Hz parsing + sliding-window segmentation.
- TCN architecture + Focal Loss for class imbalance.
- LOSO-CV outer loop; memory-efficient logging (list[dict]→DataFrame).
- Temporal post-processing pipeline.

## 3. Current best baseline (from loso_summary_20260519_165501.csv)
8 FoG subjects (S04, S10 excluded — no FoG events):

| metric | value | note |
|---|---|---|
| mean ROC-AUC | ~0.77 | range 0.39 (S08!) – 0.95 (S02) |
| mean Best-F1 | ~0.50 | **swept on test subject — optimistic, see ADR-003** |
| worst subject | S08 | AUC 0.39 (< chance) — investigate |

Caveats (do not quote these numbers in a paper as-is):
- Best-F1 uses a per-test-subject threshold → leakage (ADR-003).
- AUC-ROC overstates under 81/19 imbalance → add PR-AUC (ADR-007).

## 4. Current focus / next actions (in order)
1. [ ] Fix threshold selection (validation, frozen) and re-measure the
       post-processing gain honestly. — ADR-003
2. [ ] Confirm n_conv_per_block in basic_tcn.py; fix the RF check formula. — ADR-005
3. [ ] Make post-processing causal; report end-to-end latency. — ADR-004
4. [ ] Implement nested split + grid search (kernel_size [3,5,7], dilations) with
       the corrected RF guard. — Task C from spec
5. [ ] Label-shift to prediction, horizon = 1 s. — ADR-006
6. [ ] Add PR-AUC + episode-level metrics + detection latency. — ADR-007
7. [ ] Wire external dataset (DeFOG / O'Day) as cross-dataset test. — ADR-008

## 5. Known risks
- S08 below chance: possible label/orientation issue for that subject, or genuine
  hard case — needs a per-subject diagnostic plot before trusting the mean.
- "+4% F1" headline is at risk until ADR-003/004 are resolved.
- Switching datasets mid-project is a timeline risk — current plan keeps Daphnet
  primary (ADR-008).

## 6. Repo structure — FINALIZED (restructured 2026-06-14)
```
115Daphnet_FoG/
├── data/                  Daphnet raw SxxRyy files (untouched)
├── results/               RUN_<timestamp>/ outputs (loso_summary, curves, ...) (untouched)
├── src/
│   ├── basic_tcn.py       TCN architecture          <- need n_conv_per_block
│   ├── fog_preprocessing.py  windowing/labeling     <- need stride/overlap/label map
│   ├── trainer_tcn.py     focal loss / early stop / threshold?
│   ├── run_loso.py        LOSO + threshold + post-proc (was model/run_tcn_v2.py)  <- ADR-003/004 live here
│   └── plot_fog.py         plotting (was plot_fog.py at repo root)
├── scripts/
│   └── check_env.py       merged env check (was model/checkfile.py; check_version.py archived as duplicate)
├── _archive/
│   ├── run_tcn.py          older runner (was model/run_tcn.py)
│   └── check_version.py    superseded duplicate env-check script
├── edge/                   (empty) Jetson streaming-inference path, TBD
├── docs/
│   ├── literature/references.md  (was references.md at repo root)
│   └── deployment/         (empty) Jetson/TensorRT deployment notes, TBD
├── tests/                  (empty) split/RF/label-mapping/causality tests, TBD (R9)
├── .claude/skills/lit-review/SKILL.md  (was lit-review_SKILL.md at repo root)
├── EXPERIMENTS.md          run log (new, empty template)
├── README.md               (new, empty template)
├── requirements.txt        frozen from fog_env_64 (active venv)
├── CLAUDE.md              agent memory (this set)
├── fog_env/, fog_env_64/  venvs (ignore)
└── tree.txt
```

**What moved (2026-06-14 restructure):**
- `model/` -> `src/` (basic_tcn.py, fog_preprocessing.py, trainer_tcn.py unchanged; flat
  intra-`src/` imports still resolve because Python adds the script's own dir to
  `sys.path[0]`).
- `model/run_tcn_v2.py` -> `src/run_loso.py`; `model/run_tcn.py` -> `_archive/run_tcn.py`.
- `plot_fog.py` (root) -> `src/plot_fog.py`.
- `model/checkfile.py` (richer package-table env check) -> `scripts/check_env.py`;
  `check_version.py` (root, duplicate/simpler) -> `_archive/check_version.py`.
- `references.md` -> `docs/literature/references.md`; `lit-review_SKILL.md` ->
  `.claude/skills/lit-review/SKILL.md`.
- Path fixes (R12 — landmine paths replaced with `ROOT_DIR = Path(__file__).resolve().parents[1]`):
  - `src/plot_fog.py`: hardcoded `C:/Project/115Daphnet_FoG/data/dataset/S01R01.txt`
    -> `ROOT_DIR / "data" / "dataset" / "S01R01.txt"`; output `fog_waveform_S01R01.png`
    now written to `ROOT_DIR` explicitly.
  - `src/run_loso.py`: hardcoded `C:/Project/115Daphnet_FoG/data/dataset` ->
    `ROOT_DIR / "data" / "dataset"`; bare relative `"results"` for `run_dir` ->
    `os.path.join(ROOT_DIR, "results", ...)`.
  - `_archive/run_tcn.py` (legacy, archived) and `trainer_tcn.py`'s
    relative `best_model.pth` (CWD-relative, repo-root-relative under the new
    "run from repo root" convention) were left as-is — out of scope / not landmines
    under the new convention.
- `model/__pycache__` and `model/checkfile.py` removed (build artifact + merged file);
  `model/` directory removed once empty. `data/` and `results/` untouched.

**Smoke test (2026-06-14):** `import src.basic_tcn / src.fog_preprocessing / src.trainer_tcn`
from repo root OK (namespace package, no `__init__.py` needed). Dry-run of the
`run_loso` pipeline (1 fold: train=S02R01, val=S01R01, 1 epoch, CPU) completed
end-to-end — PASS. Only warning: pre-existing `torch.nn.utils.weight_norm`
deprecation notice (unrelated to this move; tracked separately per CLAUDE.md
engineering standards).

## 7. Roadmap (≈10 weeks to end of August)
- W1–2: lock prediction task + fix methodology (ADR-003/004/005); freeze a model.
- W3–5: Jetson Orin NX export (ONNX→TensorRT), streaming inference, IMU integration.
- W6–7: RAS closed-loop cueing (basic fixed-tempo first); measure cue lead time.
- W8–9: ablations (64 vs 32 Hz, feature channels, cross-dataset), figures, stats.
- W9–10: write-up + slides + defense prep.

## 8. To finalize next
- [x] Folder restructure (model/ -> src/, scripts/, _archive/, docs/literature/,
      docs/deployment/, edge/, tests/, EXPERIMENTS.md, README.md, requirements.txt) —
      done 2026-06-14, see §6.
- [ ] SKILL.md — repeatable procedures (run LOSO, add feature channel, grid search,
      export TensorRT, lit-review via nrp-literature MCP).
- [ ] Populate tests/ per R9 (subject-overlap, RF<=window, label-mapping, causal
      post-proc checks).
