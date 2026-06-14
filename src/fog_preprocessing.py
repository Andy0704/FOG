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

    def create_windows(self, X_numpy, y_numpy):
        # 轉換為 Tensor
        X_tensor = torch.FloatTensor(X_numpy)
        y_tensor = torch.FloatTensor(y_numpy)

        # 處理特徵 (N, C) -> 轉置 -> 展開 -> 轉置回 (batch, channels, length)
        X_transposed = X_tensor.t()
        X_windows = X_transposed.unfold(1, self.window_size, self.step_size)
        X_final = X_windows.permute(1, 0, 2)

        # 處理標籤 (N,) -> 展開 -> Any-Overlap 邏輯
        y_windows = y_tensor.unfold(0, self.window_size, self.step_size)
        # 只要視窗內有任何大於 0 的標籤，就設為 1
        y_final = (y_windows > 0).any(dim=1).float()

        return X_final, y_final
