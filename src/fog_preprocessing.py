import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class FoGPreprocessor:
    def __init__(self, window_size=64, step_size=32,
                 downsample_to_32hz=False, scale_to_g=False, add_magnitude=False):
        self.window_size = window_size
        self.step_size = step_size
        self.scaler = StandardScaler()

        # 預留開關：未來擴充用 (Task D)
        self.downsample_to_32hz = downsample_to_32hz
        self.scale_to_g = scale_to_g
        self.add_magnitude = add_magnitude

    def downsample(self, X, y, factor=2):
        """將訊號從 64Hz 降採樣至 32Hz (每隔 factor 點取一點)"""
        return X[::factor], y[::factor]

    def convert_to_g(self, X, mg_per_g=1000.0):
        """將 Daphnet 原始的 mg (milli-g) 單位轉換為 g (重力歸一化)"""
        return X / mg_per_g

    def compute_magnitude(self, X):
        """計算三軸合加速度向量 sqrt(X^2 + Y^2 + Z^2)，回傳 (N, 1)"""
        return np.linalg.norm(X, axis=1, keepdims=True)

    def apply_feature_pipeline(self, X, y):
        """依照建構子設定的開關，套用 Task D 預留的擴充處理"""
        if self.downsample_to_32hz:
            X, y = self.downsample(X, y)

        if self.scale_to_g:
            X = self.convert_to_g(X)

        if self.add_magnitude:
            magnitude = self.compute_magnitude(X)
            X = np.hstack([X, magnitude])

        return X, y

    def scale_train(self, X):
        # 讓 scaler 學習訓練集的分布，並同時進行縮放
        return self.scaler.fit_transform(X)

    def scale_val(self, X):
        return self.scaler.transform(X)

    def create_windows(self, X_numpy, y_numpy, task="detection", exclude_mask=None):
        """task="detection" (default, UNCHANGED): any-overlap window label, returns
        (X_final, y_final) -- exactly the original behavior.

        task="prediction" (ADR-014): y_numpy must already be the per-sample
        pre-FoG label from make_prediction_labels() (endpoint-based window
        label, not any-overlap). exclude_mask (per-sample bool, True=during
        freeze) is windowed the same way and any window overlapping it is
        DROPPED (not just labeled 0), since a window's content must not mix
        FoG-episode samples with a "normal" label. Returns (X_final, y_final,
        end_idx) where end_idx is the original-timeline sample index of each
        surviving window's endpoint (needed for episode-level lead-time eval).
        """
        # 轉換為 Tensor
        X_tensor = torch.FloatTensor(X_numpy)
        y_tensor = torch.FloatTensor(y_numpy)

        # 處理特徵 (N, C) -> 轉置 -> 展開 -> 轉置回 (batch, channels, length)
        X_transposed = X_tensor.t()
        X_windows = X_transposed.unfold(1, self.window_size, self.step_size)
        X_final = X_windows.permute(1, 0, 2)

        if task == "detection":
            # 處理標籤 (N,) -> 展開 -> Any-Overlap 邏輯
            y_windows = y_tensor.unfold(0, self.window_size, self.step_size)
            # 只要視窗內有任何大於 0 的標籤，就設為 1
            y_final = (y_windows > 0).any(dim=1).float()
            return X_final, y_final

        elif task == "prediction":
            n_windows = X_final.shape[0]
            end_idx = np.arange(n_windows) * self.step_size + self.window_size - 1

            # 標籤取視窗「終點樣本」的 pre-FoG 標籤 (非 any-overlap)
            y_windows = y_tensor.unfold(0, self.window_size, self.step_size)
            y_final = y_windows[:, -1].clone()

            if exclude_mask is not None:
                ex_tensor = torch.BoolTensor(np.asarray(exclude_mask))
                ex_windows = ex_tensor.unfold(0, self.window_size, self.step_size)
                keep = (~ex_windows.any(dim=1)).numpy()
            else:
                keep = np.ones(n_windows, dtype=bool)

            X_final = X_final[keep]
            y_final = y_final[keep]
            end_idx = end_idx[keep]
            return X_final, y_final, end_idx

        else:
            raise ValueError(f"Unknown task '{task}', expected 'detection' or 'prediction'.")

    @staticmethod
    def make_prediction_labels(y, horizon=64):
        """Build PREDICTION (pre-FoG) labels + exclusion mask from a single
        continuous per-file DETECTION label sequence y (0=no-freeze, 1=freeze;
        already 0-filtered by load_daphnet_file -- raw Daphnet 1=no-freeze,
        2=freeze). Must be called PER FILE, before any cross-file/cross-subject
        concatenation (ADR-014) -- concatenating first would fabricate onsets
        and pre-FoG horizon windows that straddle two unrelated recordings.

        onset = index i where y[i-1]==0 and y[i]==1 (the raw Daphnet 1->2
        transition, post 0-filtering). Positive (pre-FoG) class = the `horizon`
        samples strictly before each onset, clipped to the start of the array.
        exclude_mask marks every during-freeze sample (onset included) as True
        -- these are NEVER given a positive/negative label; we predict
        imminent onset, not ongoing freeze. If two onsets are closer together
        than `horizon`, the pre-FoG windows may overlap a preceding freeze
        episode's tail; exclude_mask always wins at window-build time
        (create_windows drops any window overlapping it), so no conflict needs
        resolving here.

        Edge case: if y[0]==1 (file starts already mid-freeze, e.g. because the
        actual 1->2 transition sample was itself 0-filtered), that leading
        freeze has no detectable onset and is NOT counted as an episode --
        its samples are still excluded via exclude_mask.

        Returns (y_pred, exclude_mask, onset_idx): same-length float32/bool
        arrays plus the int array of onset sample indices found in this file.
        """
        y = np.asarray(y)
        n = len(y)
        y_pred = np.zeros(n, dtype=np.float32)
        exclude_mask = (y == 1)

        onset_idx = np.where((y[1:] == 1) & (y[:-1] == 0))[0] + 1

        for onset in onset_idx:
            start = max(0, onset - horizon)
            y_pred[start:onset] = 1.0

        return y_pred, exclude_mask, onset_idx
