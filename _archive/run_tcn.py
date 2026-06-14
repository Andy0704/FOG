import os
import glob
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from basic_tcn import BasicTCN
from fog_preprocessing import FoGPreprocessor
from trainer_tcn import TCNTrainer
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import shutil


def load_daphnet_file(file_path):
    """讀取 Daphnet 檔案並清洗標籤"""
    # 1. 讀取純文字數據檔，處理不定長度的空白分隔
    df = pd.read_csv(file_path, sep='\s+', header=None)
    
    # 2. 依照我們剛剛討論的 iloc 切片邏輯提取資料
    X = df.iloc[:, 1:4].values  # 第 1, 2, 3 欄為腳踝 3 軸加速度
    y = df.iloc[:, -1].values   # 最後一欄為 FoG 標籤

    # 關鍵清洗：只保留標籤 1(正常) 和 2(凍結)，過濾掉 0(無效/未標記)
    valid_idx = (y == 1) | (y == 2)
    X = X[valid_idx]
    y = y[valid_idx]

    # 將標籤轉換為標準的二元分類：正常=0, 凍結=1
    y = np.where(y == 2, 1, 0)
    
    return X, y

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"目前使用的運算裝置: {device}")

    filepath = "C:/Project/115Daphnet_FoG/data/dataset"# 請確保路徑正確
    # 1. 取得資料 (優化為自動抓取)
    train_files = glob.glob(os.path.join(filepath, 'S01*.txt')) + \
                  glob.glob(os.path.join(filepath, 'S02*.txt'))
    val_files = glob.glob(os.path.join(filepath, 'S03*.txt'))

    print(f"找到 {len(train_files)} 個訓練檔, {len(val_files)} 個驗證檔")

    # 利用迴圈動態讀取與合併
    X_train_list, y_train_list = zip(*[load_daphnet_file(f) for f in train_files])
    X_train_raw = np.vstack(X_train_list)
    y_train_raw = np.concatenate(y_train_list)

    X_val_list, y_val_list = zip(*[load_daphnet_file(f) for f in val_files])
    X_val_raw = np.vstack(X_val_list)
    y_val_raw = np.concatenate(y_val_list)

    # 2. 預處理與切窗
    preprocessor = FoGPreprocessor(window_size=64, step_size=32)
    X_train_scaled = preprocessor.scale_train(X_train_raw)
    X_val_scaled = preprocessor.scale_val(X_val_raw)

    #針對特徵 X 做Z-score標準化處理
    X_train_scaled = preprocessor.scale_train(X_train_raw)
    X_val_scaled = preprocessor.scale_val(X_val_raw)
    
    X_train_windows, y_train_windows = preprocessor.create_windows(X_train_scaled, y_train_raw)
    X_val_windows, y_val_windows = preprocessor.create_windows(X_val_scaled, y_val_raw)
    
    # 3. 建立 DataLoader
    train_dataset = TensorDataset(X_train_windows, y_train_windows)
    val_dataset = TensorDataset(X_val_windows, y_val_windows)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True) # Batch size 建議調大一點如 256
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # 4. 初始化模型與訓練器
    model = BasicTCN(n_channels=3, n_classes=1)
    trainer = TCNTrainer(model, device=device)
    
    # 5. 開始訓練
    trainer.train(train_loader, val_loader, n_epochs=15)

    # 6. 畫出ROC曲線，載入最佳模型
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model.eval()

    all_probs, all_labels = [], []
    #收集整個驗證集的預測機率與真實標籤
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(trainer.device)
            outputs = model(inputs)
            # 算出機率並轉成 numpy 陣列以便 sklearn 處理
            probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    #使用 sklearn 計算 ROC 曲線數據
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    #計算約登指數 (Youden's J statistic) 找出最佳 Threshold
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]
    
    print(f" 最佳 Threshold (Youden's Index): {best_thresh:.4f}")
    
    #畫出 ROC 曲線並標記最佳點
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 隨機猜測線

    #標記最佳 Threshold 的位置
    plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', s=100, zorder=5, 
                label=f'Best Threshold = {best_thresh:.4f}')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR (Recall)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('results/roc_curve.png')
    plt.close()

    #7. 畫出 Loss 曲線
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(8, 6)) # 開啟新的畫布
    plt.plot(trainer.history['train_loss'], label='Train Loss')
    plt.plot(trainer.history['val_loss'], label='Val Loss')
    plt.title('TCN Training Curve (Focal Loss)')
    plt.legend()

    # 將時間戳記加入檔名
    loss_path = f'C:/Project/115Daphnet_FoG/results/tcn_loss_curve_{timestamp}.png'
    plt.savefig(loss_path)
    plt.close()
    print(f"📈 Loss 曲線已儲存為 {loss_path}")

    # 優化模型儲存：直接複製檔案並加上時間戳記，避免重新 serialize
    save_path2 = f'C:/Project/115Daphnet_FoG/results/tcn_fog_weights_{timestamp}.pth'
    shutil.copy('best_model.pth', save_path2)
    print(f"💾 模型權重已成功儲存至：{save_path2}")

if __name__ == "__main__":
    main()