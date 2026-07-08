import os
import glob
import json
import random
import time
import uuid
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from basic_tcn import BasicTCN
from fog_preprocessing import FoGPreprocessor
from trainer_tcn import TCNTrainer
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_curve, auc, f1_score, average_precision_score
import shutil

ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT_DIR / "results" / "_tmp_checkpoints"


def causal_median(p, k=5):
    """Past-only (causal) median filter for real-time-deployable metrics (ADR-004)."""
    return pd.Series(p).rolling(k, min_periods=1).median().to_numpy()


def causal_majority_vote(y, w=7):
    """Past-only (causal) rolling majority vote for real-time-deployable metrics (ADR-004)."""
    return (pd.Series(y).rolling(w, min_periods=1).mean() >= 0.5).astype(int).to_numpy()


def predict(model, loader, device):
    """Run model over a loader, return (probs, labels) as numpy arrays."""
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
            probs_list.extend(probs)
            labels_list.extend(labels.numpy())
    return np.array(probs_list), np.array(labels_list)

def load_daphnet_file(file_path):
    """讀取單一 Daphnet 檔案並回傳清洗後的特徵與標籤"""
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    X = df.iloc[:, 1:4].values  
    y = df.iloc[:, -1].values   
    
    #  關鍵清洗：只保留標籤 1(正常) 和 2(凍結)，過濾掉 0(無效)
    valid_idx = (y == 1) | (y == 2)
    X = X[valid_idx]
    y = y[valid_idx]
    
    # 將標籤轉換為標準二元分類：正常=0, 凍結=1
    y = np.where(y == 2, 1, 0)

    return X, y


def discover_subject_files(root_dir):
    """Scan data/dataset/*.txt and group files by their 3-char subject ID
    (e.g. 'S01R01.txt' -> 'S01'). Returns {subject_id: [file_path, ...]}."""
    filepath = root_dir / "data" / "dataset"
    all_files = glob.glob(os.path.join(filepath, "*.txt"))
    subject_files = {}
    for f in all_files:
        subj_id = os.path.basename(f)[:3]
        subject_files.setdefault(subj_id, []).append(f)
    return subject_files


def get_fog_subjects(subject_files):
    """FOG_SUBJECTS: all discovered subjects except S04/S10 (no FoG events,
    excluded from LOSO per ADR-002)."""
    subjects = sorted(subject_files.keys())
    return sorted([s for s in subjects if s not in ("S04", "S10")])


def build_xy(subj_list, subject_files):
    """Concatenate raw (filtered, 0/1-remapped) X, y across a list of subjects."""
    X_list, y_list = [], []
    for subj in subj_list:
        for f in subject_files[subj]:
            X, y = load_daphnet_file(f)
            X_list.append(X)
            y_list.append(y)
    return np.vstack(X_list), np.concatenate(y_list)


def build_prediction_records(subj_list, subject_files, horizon):
    """Per-file (NOT concatenated) PREDICTION-task label builder (ADR-014).
    Iterates subject_files[subj] file-by-file so onset detection and the
    pre-FoG horizon window never straddle two different recordings/subjects --
    unlike build_xy(), which vstacks/concatenates raw arrays across files."""
    records = []
    for subj in subj_list:
        for f in subject_files[subj]:
            X, y = load_daphnet_file(f)
            y_pred, exclude_mask, onset_idx = FoGPreprocessor.make_prediction_labels(y, horizon)
            records.append({
                'file': f, 'subj': subj, 'X_raw': X, 'y_pred': y_pred,
                'exclude_mask': exclude_mask, 'onset_idx': onset_idx,
            })
    return records


def windows_from_records(records, preprocessor, scale_fn):
    """Scale (via scale_fn, an already-fit transform) and window each record's
    file individually (task='prediction'), then concatenate. Tracks each
    surviving window's original endpoint sample index and source-file id
    (index into `records`) for episode-level lead-time evaluation."""
    X_parts, y_parts, end_idx_parts, file_id_parts = [], [], [], []
    for fid, rec in enumerate(records):
        if len(rec['X_raw']) < preprocessor.window_size:
            continue
        X_scaled = scale_fn(rec['X_raw'])
        X_w, y_w, end_idx = preprocessor.create_windows(
            X_scaled, rec['y_pred'], task="prediction", exclude_mask=rec['exclude_mask'])
        if len(y_w) == 0:
            continue
        X_parts.append(X_w)
        y_parts.append(y_w)
        end_idx_parts.append(end_idx)
        file_id_parts.append(np.full(len(y_w), fid, dtype=int))

    if not X_parts:
        return torch.empty(0, 3, preprocessor.window_size), torch.empty(0), \
            np.array([], dtype=int), np.array([], dtype=int)

    X_all = torch.cat(X_parts, dim=0)
    y_all = torch.cat(y_parts, dim=0)
    end_idx_all = np.concatenate(end_idx_parts)
    file_id_all = np.concatenate(file_id_parts)
    return X_all, y_all, end_idx_all, file_id_all


def print_prediction_balance(label, y_windows):
    """Print pre-FoG window class balance (PREDICTION-task diagnostic); flags
    subjects with very few positive windows."""
    y_np = y_windows.numpy() if torch.is_tensor(y_windows) else np.asarray(y_windows)
    n = len(y_np)
    n_pos = int((y_np == 1).sum())
    n_neg = n - n_pos
    pct = (100.0 * n_pos / n) if n else float('nan')
    print(f"[PREDICTION class balance] {label}: total={n} windows, "
          f"pos(pre-FoG)={n_pos} ({pct:.2f}%), neg={n_neg}")
    if n_pos < 20:
        print(f"[WARN] {label}: only {n_pos} positive (pre-FoG) windows -- "
              f"may be too sparse to learn a stable decision boundary.")
    return n_pos, n_neg


def print_exclusion_summary(records):
    """Confirm the freeze-period exclusion mask actually removed samples, and
    report how many onsets were detected (ADR-014 verifiability)."""
    total_samples = sum(len(r['y_pred']) for r in records)
    total_excluded = sum(int(r['exclude_mask'].sum()) for r in records)
    total_onsets = sum(len(r['onset_idx']) for r in records)
    pct = (100.0 * total_excluded / total_samples) if total_samples else float('nan')
    print(f"[PREDICTION exclusion] files={len(records)}, total_samples={total_samples}, "
          f"excluded_during-freeze_samples={total_excluded} ({pct:.2f}%), "
          f"onsets_detected={total_onsets}")


def episode_level_metrics(test_records, y_pred_binary, end_idx_all, file_id_all, horizon, fs=64.0):
    """Per FoG onset in the TEST subject: recall = was there >=1 positive
    prediction among the windows whose endpoint falls in the true pre-FoG
    horizon [onset-horizon, onset-1); lead_time_s = (onset - endpoint of the
    EARLIEST positive window) / fs, computed only for onsets with a hit.
    Onsets with zero valid pre-FoG windows (too close to file start, or fully
    swallowed by a preceding excluded freeze) are skipped and counted."""
    recalls = []
    lead_times_s = []
    n_skipped_no_window = 0
    for fid, rec in enumerate(test_records):
        for onset in rec['onset_idx']:
            in_episode = (file_id_all == fid) & (end_idx_all >= onset - horizon) & (end_idx_all < onset)
            if not in_episode.any():
                n_skipped_no_window += 1
                continue
            ep_preds = np.asarray(y_pred_binary)[in_episode]
            ep_end_idx = end_idx_all[in_episode]
            hit = bool(ep_preds.any())
            recalls.append(1 if hit else 0)
            if hit:
                first_end_idx = ep_end_idx[ep_preds.astype(bool)].min()
                lead_times_s.append((onset - first_end_idx) / fs)
    return recalls, lead_times_s, n_skipped_no_window


def inner_val_score(va_probs, va_labels):
    """Selection metric for choosing val_subj among inner LOSO candidates:
    ROC-AUC on causal-median(k=5)-smoothed val probs (ADR-003 Option 1).
    Distinct from the per-epoch early-stopping criterion (val focal-loss,
    unchanged) -- this only decides WHICH candidate's split gets reused
    for the final outer retrain (or, for Optuna tuning, contributes directly
    to the val-only tuning objective -- see tune_optuna.py)."""
    if len(np.unique(va_labels)) < 2:
        return float('nan')
    va_smoothed = causal_median(va_probs, k=5)
    fpr, tpr, _ = roc_curve(va_labels, va_smoothed)
    return auc(fpr, tpr)


def run_split(train_subjs, val_subj, test_subj, subject_files, device, n_epochs,
              window_size=64, step_size=32, kernel_size=5, dilations=None, dropout=0.2,
              alpha=0.75, gamma=2.0, lr=0.0001, weight_decay=0.0001, batch_size=256,
              split_timings=None, verbose=True, task="detection", horizon=64):
    """Train one fresh model: train on train_subjs, early-stop on val_subj
    (per-epoch stopping criterion is still val focal-loss, unchanged in
    trainer_tcn.py). Returns (model, trainer, preprocessor, va_probs,
    va_labels, elapsed_s, epochs_run).

    HARD CONSTRAINT (ADR-003): test_subj must never appear in train_subjs
    or be used as val_subj, in any inner or outer call. test_subj is used
    ONLY for this leakage assertion -- its data is never loaded here.

    kernel_size/dilations/dropout/alpha/gamma/lr/weight_decay/batch_size are
    tunable (ADR-013, src/tune_optuna.py); defaults reproduce the existing
    baseline (RUN_20260702_202440) exactly.

    task="detection" (default): UNCHANGED behavior -- build_xy() concatenates
    raw X/y across train_subjs/val_subj, create_windows() any-overlap labels.
    task="prediction" (ADR-014 label-shift, see fog_preprocessing.make_
    prediction_labels): build_prediction_records() builds per-file pre-FoG
    labels + exclusion masks (never concatenating raw y across files first),
    the scaler is fit once on the full concatenated train pool, then each
    file is windowed individually via windows_from_records() so windows never
    straddle two files.
    """
    assert test_subj not in train_subjs, \
        f"LEAKAGE: test_subj {test_subj} found in train_subjs {train_subjs}!"
    assert val_subj not in train_subjs and val_subj != test_subj, \
        f"LEAKAGE: val_subj {val_subj} overlaps train ({train_subjs}) or test ({test_subj})!"

    preprocessor = FoGPreprocessor(window_size=window_size, step_size=step_size)

    if task == "detection":
        X_train_raw, y_train_raw = build_xy(train_subjs, subject_files)
        X_val_raw, y_val_raw = build_xy([val_subj], subject_files)

        # 預處理與切窗：scaler 只在 TRAIN 上 fit (S1)
        X_train_scaled = preprocessor.scale_train(X_train_raw)
        X_val_scaled = preprocessor.scale_val(X_val_raw)

        X_train_windows, y_train_windows = preprocessor.create_windows(X_train_scaled, y_train_raw)
        X_val_windows, y_val_windows = preprocessor.create_windows(X_val_scaled, y_val_raw)

    elif task == "prediction":
        train_records = build_prediction_records(train_subjs, subject_files, horizon)
        val_records = build_prediction_records([val_subj], subject_files, horizon)

        # scaler 只在 TRAIN 上 fit (S1) -- fit once on the full concatenated
        # train pool, then apply (transform-only) per file below.
        X_train_raw_all = np.vstack([r['X_raw'] for r in train_records])
        preprocessor.scale_train(X_train_raw_all)

        X_train_windows, y_train_windows, _, _ = windows_from_records(
            train_records, preprocessor, preprocessor.scale_val)
        X_val_windows, y_val_windows, _, _ = windows_from_records(
            val_records, preprocessor, preprocessor.scale_val)

        if verbose:
            print_prediction_balance(f"train(subjs={train_subjs})", y_train_windows)
            print_prediction_balance(f"val={val_subj}", y_val_windows)
            print_exclusion_summary(train_records + val_records)
    else:
        raise ValueError(f"Unknown task '{task}', expected 'detection' or 'prediction'.")

    train_loader = DataLoader(TensorDataset(X_train_windows, y_train_windows), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_windows, y_val_windows), batch_size=batch_size, shuffle=False)

    # 模型初始化 (每一折/每個候選都要全新的模型，避免資料污染！)
    # BasicTCN.__init__ asserts RF <= window_size and prints [RF] ... (ADR-005) --
    # unconditional, runs for every model regardless of caller.
    model = BasicTCN(n_channels=3, n_classes=1, kernel_size=kernel_size,
                      dropout=dropout, dilations=dilations, window_size=window_size)
    # Unique per-call checkpoint path (ADR-013): a shared fixed 'best_model.pth' in
    # CWD is unsafe once architecture varies across calls (Optuna) or two processes
    # train concurrently against the same repo -- either can silently load a
    # mismatched checkpoint. Caller is responsible for moving/removing this path
    # once done with it (see main()'s backup step and tune_optuna.py's cleanup).
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = str(CHECKPOINT_DIR / f"ckpt_{os.getpid()}_{uuid.uuid4().hex[:8]}.pth")
    trainer = TCNTrainer(model, device=device, alpha=alpha, gamma=gamma, lr=lr,
                          weight_decay=weight_decay, checkpoint_path=checkpoint_path)

    if verbose:
        print(f" Train: {len(y_train_windows)} windows, Val({val_subj}): {len(y_val_windows)} windows...")
    t0 = time.time()
    trainer.train(train_loader, val_loader, n_epochs=n_epochs)
    elapsed = time.time() - t0
    epochs_run = len(trainer.history['train_loss'])
    if split_timings is not None:
        split_timings.append((elapsed, epochs_run))

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=trainer.device, weights_only=True))
        model.eval()
        if verbose:
            print(f"Loaded best checkpoint (early-stopped on val={val_subj})")

    va_probs, va_labels = predict(model, val_loader, trainer.device)
    return model, trainer, preprocessor, va_probs, va_labels, elapsed, epochs_run, checkpoint_path


def main(n_epochs=50, subjects_to_run=None, seed=42, kernel_size=5, dilations=None,
         dropout=0.2, alpha=0.75, gamma=2.0, lr=0.0001, weight_decay=0.0001, batch_size=256,
         task="detection", horizon=64):
    if task not in ("detection", "prediction"):
        raise ValueError(f"Unknown task '{task}', expected 'detection' or 'prediction'.")
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"[SEED] {SEED}")
    if task == "detection":
        print("[TASK] DETECTION (label at t, horizon=n/a)")
    else:
        print(f"[TASK] PREDICTION (label shifted +H before onset, horizon={horizon} "
              f"samples = {horizon/64:.2f}s @ 64Hz; ADR-014)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE] {device}")

    subject_files = discover_subject_files(ROOT_DIR)
    if not subject_files:
        print(" 找不到任何 .txt 檔案，請檢查資料夾路徑！")
        return

    # 建立本次實驗的專屬資料夾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ROOT_DIR, "results", f"RUN_{timestamp}")
    models_dir = os.path.join(run_dir, "models")
    curves_dir = os.path.join(run_dir, "curves")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)

    print(f"實驗資料夾已建立: {run_dir}")

    # Crash-safety: resolved up front so each outer fold can append its row to
    # the summary CSV immediately (not only at the very end of the run).
    summary_path = os.path.join(run_dir, f"loso_summary_{timestamp}.csv")

    def append_result_row(row):
        """Append one fold's result row to summary_path immediately, so a
        mid-run crash keeps every fold finished so far. Writes the header
        only on the first row."""
        row_df = pd.DataFrame([row])
        write_header = not os.path.exists(summary_path)
        row_df.to_csv(summary_path, mode='a', header=write_header, index=False)

    def fmt_mean_std(series):
        """Mean +/- std, guarding the n<2 case (pandas .std() is NaN for a
        single sample) so the summary never prints '± nan'."""
        n = len(series)
        if n == 0:
            return "n/a (n=0)"
        mean = series.mean()
        if n < 2:
            return f"{mean:.4f} +/- n/a (n=1)"
        return f"{mean:.4f} +/- {series.std():.4f}"

    # 1. 依照受試者 (Subject) 分組檔案 (假設檔名格式為 S01R01.txt，取前 3 個字元 'S01' 當作 ID)
    subjects = sorted(subject_files.keys())
    print(f"找到 {len(subjects)} 位受試者: {subjects}")

    # S04/S10 無 FoG 事件 (ADR-002)，排除於 LOSO 之外。
    # TODO: 之後另建 FAR (False-Alarm-Rate) 套件評估 S04/S10 — 本次不實作。
    FOG_SUBJECTS = get_fog_subjects(subject_files)
    print(f"FoG 受試者 (LOSO 迴圈): {FOG_SUBJECTS}")

    test_subjects = FOG_SUBJECTS
    if subjects_to_run is not None:
        test_subjects = [s for s in FOG_SUBJECTS if s in subjects_to_run]
        print(f"本次僅執行 (test_subj): {test_subjects}")

    # 2. 開始 LOSO 大迴圈 (巢狀 train/val/test 三分割，見 ADR-003)
    loso_results = []
    all_predictions_list = [] #用於存取詳細機率 (TEST 受試者)
    split_timings = []  # (elapsed_s, epochs_run) for every model trained this call -- used
                         # for the wall-clock estimate printed at the end (ADR-003)

    # build_xy / run_split / inner_val_score are module-level (see above) so
    # tune_optuna.py can reuse them directly instead of reimplementing (ADR-013).
    def _run_split(train_subjs, val_subj, test_subj):
        return run_split(train_subjs, val_subj, test_subj, subject_files, device, n_epochs,
                          kernel_size=kernel_size, dilations=dilations, dropout=dropout,
                          alpha=alpha, gamma=gamma, lr=lr, weight_decay=weight_decay,
                          batch_size=batch_size, split_timings=split_timings,
                          task=task, horizon=horizon)

    n_outer_folds = len(test_subjects)
    for outer_idx, test_subj in enumerate(test_subjects, start=1):
        print(f"\n{'='*30}")
        print(f"開始執行 LOSO Fold: 測試受試者為 【{test_subj}】")
        print(f"{'='*30}")

        pool = [s for s in FOG_SUBJECTS if s != test_subj]

        # --- INNER LOSO (ADR-003 Option 1): rotate val_subj over the pool,
        # score each candidate by inner_val_score(), select the MEDIAN scorer
        # (avoids picking whichever val_subj happened to be luckiest). This
        # replaces the old fixed `val_subj = pool[0]`. OPTION 2 (ensemble
        # across all 7 inner-split models instead of retraining once) is NOT
        # implemented -- see TODO below.
        print(f"--- Inner LOSO: rotating val_subj over {pool} (test={test_subj} excluded) ---")
        inner_scores = {}
        for candidate in pool:
            inner_train_subjs = [s for s in pool if s != candidate]
            assert test_subj not in inner_train_subjs and candidate != test_subj, \
                f"LEAKAGE: test_subj {test_subj} touched by inner split (candidate={candidate})!"
            _, _, _, va_probs, va_labels, elapsed, epochs_run, inner_ckpt = _run_split(
                inner_train_subjs, candidate, test_subj)
            if os.path.exists(inner_ckpt):
                os.remove(inner_ckpt)  # inner candidates are throwaway, don't keep the checkpoint
            score = inner_val_score(va_probs, va_labels)
            inner_scores[candidate] = score
            print(f"[inner val={candidate}] ROC-AUC(smoothed)={score:.4f}  "
                  f"wall-clock={elapsed:.1f}s  epochs_run={epochs_run}")

        valid_scores = {k: v for k, v in inner_scores.items() if not np.isnan(v)}
        if not valid_scores:
            raise RuntimeError(
                f"All inner val candidates for test={test_subj} have single-class "
                f"labels; cannot select val_subj via ROC-AUC.")
        ranked = sorted(valid_scores.items(), key=lambda kv: kv[1])
        val_subj, median_score = ranked[len(ranked) // 2]
        print(f"Selected val_subj (median inner ROC-AUC rule): {val_subj} (score={median_score:.4f})")
        inner_scores_json = json.dumps({
            k: (None if np.isnan(v) else round(float(v), 4)) for k, v in inner_scores.items()
        })

        # TODO(ADR-003 Option 2, not implemented): instead of retraining once
        # on the median-selected val_subj below, keep all 7 inner-split
        # models and ensemble their TEST-set probabilities (or their frozen
        # thresholds) for a more robust test estimate. Would require keeping
        # per-inner-candidate checkpoints instead of discarding them.

        train_subjs = [s for s in pool if s != val_subj]
        print(f"Train subjects: {train_subjs}")
        print(f"Val subject:    {val_subj} (selected via inner LOSO)")
        print(f"Test subject:   {test_subj}")
        assert test_subj not in train_subjs and val_subj not in train_subjs and val_subj != test_subj, \
            "Subject leakage between train/val/test splits!"

        # --- OUTER final split: retrain once on train_subjs, early-stop on
        # the selected val_subj. This is the model used for TEST evaluation.
        model, trainer, preprocessor, va_probs, va_labels, elapsed, epochs_run, final_ckpt = _run_split(
            train_subjs, val_subj, test_subj)
        print(f"Final retrain done (train={train_subjs}, val={val_subj}) "
              f"wall-clock={elapsed:.1f}s epochs_run={epochs_run}")

        # --- Build TEST loader (task-dependent construction; downstream
        # metric code below is shared/unchanged between tasks). ---
        test_records = None
        test_end_idx = test_file_id = None
        if task == "detection":
            X_test_raw, y_test_raw = build_xy([test_subj], subject_files)
            X_test_scaled = preprocessor.scale_val(X_test_raw)
            X_test_windows, y_test_windows = preprocessor.create_windows(X_test_scaled, y_test_raw)
        else:  # task == "prediction"
            test_records = build_prediction_records([test_subj], subject_files, horizon)
            X_test_windows, y_test_windows, test_end_idx, test_file_id = windows_from_records(
                test_records, preprocessor, preprocessor.scale_val)
            print_prediction_balance(f"TEST={test_subj}", y_test_windows)
            print_exclusion_summary(test_records)
        test_loader = DataLoader(TensorDataset(X_test_windows, y_test_windows), batch_size=batch_size, shuffle=False)

        # 6. 繪製並儲存 Loss Curve (最終 retrain 的曲線，用於 TEST 評估) ---
        plt.figure(figsize=(10, 5))
        plt.plot(trainer.history['train_loss'], label='Train Loss')
        plt.plot(trainer.history['val_loss'], label='Val Loss')
        plt.title(f'Loss Curve - Subject {test_subj}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        curve_path = os.path.join(curves_dir, f"loss_{test_subj}.png")
        plt.savefig(curve_path)
        plt.close() # 務必關閉畫布

        # 9. 在 TEST 受試者上取得機率 (最終評估)
        # (va_probs/va_labels already computed inside run_split() above, from
        # this same final-retrain model's val_loader.)
        te_probs, te_labels = predict(model, test_loader, trainer.device)

        # 製作 TEST 受試者的詳細預測表
        subj_df = pd.DataFrame({
            'Subject': test_subj,
            'Window_Idx': range(len(te_probs)),
            'True_Label': te_labels,
            'Pred_Prob': te_probs
        })
        all_predictions_list.append(subj_df)

        # 若 VAL 或 TEST 只有單一類別標籤，ROC/PR-AUC 無法計算
        if len(np.unique(va_labels)) > 1 and len(np.unique(te_labels)) > 1:

            # --- Threshold 選擇：只在 VAL 上做 (S2) ---
            va_smoothed = causal_median(va_probs, k=5)
            fpr, tpr, thresholds = roc_curve(va_labels, va_smoothed)
            best_idx = np.argmax(tpr - fpr)
            best_thresh = thresholds[best_idx]
            print(f"【{test_subj}】 (val={val_subj}) 凍結 Threshold: {best_thresh:.4f}")

            # --- 套用凍結 Threshold 於 TEST (S2/S4: 因果後處理) ---
            te_smoothed = causal_median(te_probs, k=5)
            y_pred = causal_majority_vote((te_smoothed >= best_thresh).astype(int), w=7)

            test_f1 = f1_score(te_labels, y_pred)
            test_pr_auc = average_precision_score(te_labels, te_smoothed)
            fpr_t, tpr_t, _ = roc_curve(te_labels, te_smoothed)
            test_roc_auc = auc(fpr_t, tpr_t)

            print(f"【{test_subj}】 Test_F1: {test_f1:.4f}, Test_PR_AUC: {test_pr_auc:.4f}, "
                  f"Test_ROC_AUC: {test_roc_auc:.4f}")

            episode_recall = episode_mean_lead_s = episode_n = None
            if task == "prediction":
                recalls, lead_times_s, n_skipped = episode_level_metrics(
                    test_records, y_pred, test_end_idx, test_file_id, horizon)
                episode_n = len(recalls)
                episode_recall = (sum(recalls) / episode_n) if episode_n else float('nan')
                episode_mean_lead_s = (sum(lead_times_s) / len(lead_times_s)) if lead_times_s else float('nan')
                print(f"[EPISODE-LEVEL] {test_subj}: n_episodes_evaluated={episode_n} "
                      f"(skipped_no_window={n_skipped}), recall={episode_recall:.4f}, "
                      f"mean_lead_time_s={episode_mean_lead_s:.4f} (n_hits={len(lead_times_s)})")

            # 繪製 TEST ROC Curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_t, tpr_t, label=f'AUC = {test_roc_auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--') # 加上對角線 (基準線)
            plt.title(f'ROC Curve (Test) - {test_subj}')
            plt.legend()
            save_path = os.path.join(run_dir, "curves", f"roc_{test_subj}.png")
            plt.savefig(save_path)
            plt.close()
            print(f" 已儲存 ROC 圖: {save_path}")

            # 儲存該折的結果 (寫入記憶體 list + 立即 append 到 CSV, 崩潰安全)
            result_row = {
                'Subject': test_subj,
                'Val_Subject_selected': val_subj,
                'Best_Threshold': best_thresh,
                'Test_F1': test_f1,
                'Test_PR_AUC': test_pr_auc,
                'Test_ROC_AUC': test_roc_auc,
                'inner_scores_json': inner_scores_json
            }
            if task == "prediction":
                result_row['Episode_N'] = episode_n
                result_row['Episode_Recall'] = episode_recall
                result_row['Episode_MeanLeadTime_s'] = episode_mean_lead_s
            loso_results.append(result_row)
            append_result_row(result_row)

            # 備份該折的最佳模型
            target_path = os.path.join(models_dir, f"best_model_{test_subj}.pth")
            shutil.move(final_ckpt, target_path)

            print(f"[OUTER {outer_idx}/{n_outer_folds}] test={test_subj} done, "
                  f"val={val_subj}, ROC-AUC={test_roc_auc:.4f}")
        else:
            print(f"【{test_subj}】 VAL 或 TEST 只有單一類別標籤，無法計算 ROC/PR-AUC。")
            result_row = {
                'Subject': test_subj,
                'Val_Subject_selected': val_subj,
                'Best_Threshold': np.nan,
                'Test_F1': np.nan,
                'Test_PR_AUC': np.nan,
                'Test_ROC_AUC': np.nan,
                'inner_scores_json': inner_scores_json
            }
            loso_results.append(result_row)
            append_result_row(result_row)
            # 處理掉殘留的 checkpoint, 避免下一輪混淆
            if os.path.exists(final_ckpt):
                os.remove(final_ckpt)

            print(f"[OUTER {outer_idx}/{n_outer_folds}] test={test_subj} done, "
                  f"val={val_subj}, ROC-AUC=nan (single-class val or test)")


    # 7. 總結所有 LOSO 結果
    print("\n" + "="*60)
    print(" LOSO 交叉驗證最終總結報告")
    print("="*60)
    
    # 最終總結報告 (summary_path 已在迴圈中逐折 append, 這裡不再重寫檔案, 只彙整記憶體結果)
    results_df = pd.DataFrame(loso_results)
    print(f"\n LOSO 總結報告已逐折儲存 (crash-safe incremental append): {summary_path}")

    valid_f1 = results_df['Test_F1'].dropna()
    valid_pr_auc = results_df['Test_PR_AUC'].dropna()
    valid_roc_auc = results_df['Test_ROC_AUC'].dropna()

    print("\n 整體平均表現 (frozen-threshold, TEST subject only):")
    if not valid_f1.empty:
        print(f" Mean Test_F1:      {fmt_mean_std(valid_f1)}")
        print(f" Mean Test_PR_AUC:  {fmt_mean_std(valid_pr_auc)}")
        print(f" Mean Test_ROC_AUC: {fmt_mean_std(valid_roc_auc)}")
    
    # 儲存詳細機率大表
    if all_predictions_list:
        all_details_df = pd.concat(all_predictions_list, axis=0, ignore_index=True)
        details_filename = os.path.join(run_dir, f'loso_detailed_predictions_{timestamp}.csv')
        all_details_df.to_csv(details_filename, index=False)
        print(f"詳細預測機率已存至: {details_filename}")

    # 8. Wall-clock estimate for a full 8-fold x 50-epoch inner-LOSO run,
    # extrapolated from the models actually trained THIS call (ADR-003).
    # Cost model: 8 outer folds x 8 splits/fold (7 inner + 1 final retrain).
    if split_timings:
        total_elapsed = sum(e for e, _ in split_timings)
        total_epochs = sum(ep for _, ep in split_timings)
        n_splits_observed = len(split_timings)
        avg_time_per_split = total_elapsed / n_splits_observed
        avg_time_per_epoch = (total_elapsed / total_epochs) if total_epochs else float('nan')
        avg_epochs_run = total_epochs / n_splits_observed

        FULL_OUTER_FOLDS = 8
        SPLITS_PER_FOLD = 8  # 7 inner candidates + 1 final retrain
        FULL_N_EPOCHS = 50

        upper_bound_s = avg_time_per_epoch * FULL_N_EPOCHS * SPLITS_PER_FOLD * FULL_OUTER_FOLDS
        capped_at_n_epochs = (avg_epochs_run >= n_epochs - 1e-9)

        print("\n" + "=" * 60)
        print(" WALL-CLOCK ESTIMATE (ADR-003 inner-LOSO cost projection)")
        print("=" * 60)
        print(f"Observed this run: {n_splits_observed} model(s) trained, "
              f"n_epochs cap={n_epochs}, total wall-clock={total_elapsed:.1f}s")
        print(f"Avg wall-clock per split: {avg_time_per_split:.1f}s "
              f"(avg {avg_epochs_run:.1f} epochs run per split)")
        print(f"Avg wall-clock per epoch: {avg_time_per_epoch:.2f}s")
        if capped_at_n_epochs:
            print(f"NOTE: splits ran the full n_epochs={n_epochs} cap without early "
                  f"stopping triggering -- this smoke UNDER-estimates whether early "
                  f"stopping would shorten a real n_epochs={FULL_N_EPOCHS} run.")
        print(f"Projected FULL RUN ({FULL_OUTER_FOLDS} outer folds x {SPLITS_PER_FOLD} "
              f"splits/fold x {FULL_N_EPOCHS} epochs, upper bound / no early stopping):")
        print(f"  {upper_bound_s:.0f}s  (~{upper_bound_s/60:.1f} min, ~{upper_bound_s/3600:.2f} hr)")
        print("Assumes linear scaling with epoch count and similar per-split cost across "
              "subjects/folds; actual time will likely be LOWER once patience=5 early "
              "stopping triggers on most splits, and may vary with per-subject data volume.")

if __name__ == "__main__":
    main()


# 首先將WARNING MESSAGE消除
# 目前進度 ==Week 3~4 edge==
# =============================================================================
# 近期實作目標:
# 1. 機率平滑化(post-processing smoothing): 加入movering average/ median filter
#    預計顯著提升 F1-SCORE
# 2. 擴展receptive field: 之前設定的 kernel_size 和 dilation_size 涵蓋範圍不夠大
# 3. 加入合加速度向量
#    看整體的運動強度，增加對不同感測器配戴角度的強健性
