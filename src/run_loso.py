import os
import glob
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
from sklearn.metrics import roc_curve, auc, f1_score
import shutil
import scipy.signal as signal

ROOT_DIR = Path(__file__).resolve().parents[1]

def load_daphnet_file(file_path):
    """讀取單一 Daphnet 檔案並回傳清洗後的特徵與標籤"""
    df = pd.read_csv(file_path, sep='\s+', header=None)
    X = df.iloc[:, 1:4].values  
    y = df.iloc[:, -1].values   
    
    #  關鍵清洗：只保留標籤 1(正常) 和 2(凍結)，過濾掉 0(無效)
    valid_idx = (y == 1) | (y == 2)
    X = X[valid_idx]
    y = y[valid_idx]
    
    # 將標籤轉換為標準二元分類：正常=0, 凍結=1
    y = np.where(y == 2, 1, 0)
    
    return X, y

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" 目前使用的運算裝置: {device}")
    
    filepath = ROOT_DIR / "data" / "dataset"
    all_files = glob.glob(os.path.join(filepath, "*.txt"))
    
    if not all_files:
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

    # 1. 依照受試者 (Subject) 分組檔案
    # 假設檔名格式為 S01R01.txt，我們取前 3 個字元 'S01' 當作 ID
    subject_files = {}
    for f in all_files:
        subj_id = os.path.basename(f)[:3] 
        if subj_id not in subject_files:
            subject_files[subj_id] = []
        subject_files[subj_id].append(f)
        
    subjects = sorted(list(subject_files.keys()))
    print(f"找到 {len(subjects)} 位受試者: {subjects}")
    

    # 2. 開始 LOSO 大迴圈
    loso_results = []
    all_predictions_list = [] #用於存取詳細機率

    for test_subj in subjects:
        print(f"\n{'='*30}")
        print(f"開始執行 LOSO Fold: 測試受試者為 【{test_subj}】")
        print(f"{'='*30}")
        
        train_X_list, train_y_list = [], []
        val_X_list, val_y_list = [], []
        
        # 準備該折 (Fold) 的訓練集與測試集
        for subj, files in subject_files.items():
            for f in files:
                X, y = load_daphnet_file(f)
                if subj == test_subj:
                    val_X_list.append(X)
                    val_y_list.append(y)
                else:
                    train_X_list.append(X)
                    train_y_list.append(y)
                    
        # 如果該受試者沒有有效資料 (例如全都是標籤 0 被濾掉)，則跳過
        if not val_X_list:
            print(f"受試者 {test_subj} 沒有有效資料，跳過此折。")
            continue

        X_train_raw = np.vstack(train_X_list)
        y_train_raw = np.concatenate(train_y_list)
        X_val_raw = np.vstack(val_X_list)
        y_val_raw = np.concatenate(val_y_list)
        
        # 3. 預處理與切窗
        preprocessor = FoGPreprocessor(window_size=64, step_size=32)
        X_train_scaled = preprocessor.scale_train(X_train_raw)
        X_val_scaled = preprocessor.scale_val(X_val_raw)
        
        X_train_windows, y_train_windows = preprocessor.create_windows(X_train_scaled, y_train_raw)
        X_val_windows, y_val_windows = preprocessor.create_windows(X_val_scaled, y_val_raw)
        
        # 建立 DataLoader
        train_loader = DataLoader(TensorDataset(X_train_windows, y_train_windows), batch_size=256, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_windows, y_val_windows), batch_size=256, shuffle=False)
        
        # 4. 模型初始化 (非常重要：每一折都要全新的模型，避免資料污染！)
        model = BasicTCN(n_channels=3, n_classes=1)
        trainer = TCNTrainer(model, device=device)
        
        # 5. 開始訓練該折
        print(f" 開始訓練 (Train: {len(y_train_windows)} 窗, Test: {len(y_val_windows)} 窗)...")
        trainer.train(train_loader, val_loader, n_epochs=50)

        # 6. 繪製並儲存 Loss Curve ---
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

        # 7. 從根目錄載入最佳模型並計算 F1/ROC 與 Threshold
        if os.path.exists('best_model.pth'):
            model.load_state_dict(torch.load('best_model.pth'))
            model.eval()
            print(f"已載入 {test_subj} 的最佳模型進行評估")
        
        # 進行預測取得機率
        subj_probs = []
        subj_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(trainer.device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
                subj_probs.extend(probs)
                subj_labels.extend(labels.numpy())

        # 製作該受試者的詳細預測表
        subj_df = pd.DataFrame({
            'Subject': test_subj,
            'Window_Idx': range(len(subj_probs)),
            'True_Label': subj_labels,
            'Pred_Prob': subj_probs
        })
        all_predictions_list.append(subj_df)

        # 如果該受試者完全沒有發生 FoG (全為正常)，ROC 會報錯，需加個保護機制(len>1)
        if len(np.unique(subj_labels)) > 1:

            # --- 1. 中位數濾波 (消除單點雜訊，保留邊緣) ---
            subj_probs_smoothed = signal.medfilt(np.array(subj_probs), kernel_size=5)

            fpr, tpr, thresholds = roc_curve(subj_labels, subj_probs_smoothed)
        
             # 確認 fpr, tpr是否資料缺失，準備繪製 ROC Curve
            print(f"DEBUG: {test_subj} ROC 數據長度 -> FPR: {len(fpr)}, TPR: {len(tpr)}")
            roc_auc = auc(fpr, tpr)
            if len(fpr) > 0 and len(tpr) > 0:
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
                plt.plot([0, 1], [0, 1], 'k--') # 加上對角線 (基準線)
                plt.title(f'ROC Curve - {test_subj}')
                plt.legend()

                # 確保儲存路徑是存在的
                save_path = os.path.join(run_dir, "curves", f"roc_{test_subj}.png")
                plt.savefig(save_path)
                plt.close()
                print(f" 已儲存 ROC 圖: {save_path}")
            else:
                print(f" 無法繪圖，因為 fpr/tpr 為空")

            J = tpr - fpr
            best_idx = np.argmax(J)
            best_thresh = thresholds[best_idx]

            # --- 2. 二元化: 使用平滑後的機率與最佳閾值進行切分
            y_pred = (subj_probs_smoothed >= best_thresh).astype(int)

            # --- 3. Major Voting多數決機制: 加在求出y_pred之後，計算指標之前
            y_pred_voted = pd.Series(y_pred).rolling(window=7, center=True).median().fillna(0).astype(int).values
            best_f1 = f1_score(subj_labels, y_pred_voted)
            print(f"【{test_subj}】 最佳 Threshold: {best_thresh:.4f} 對應 AUC: {roc_auc:.4f}, F1: {best_f1:.4f}")
            
            # 儲存該折的結果
            loso_results.append({
                'Subject': test_subj, 
                'AUC': roc_auc, 
                'Best_Threshold': best_thresh,
                'Best_F1': best_f1
            })
            
            # 備份該折的最佳模型
            target_path = os.path.join(models_dir, f"best_model_{test_subj}.pth")
            shutil.move('best_model.pth', target_path)
        else:
            print(f"【{test_subj}】 只有單一類別標籤 (可能皆未發病)，無法計算 ROC AUC。")
            loso_results.append({'Subject': test_subj, 'AUC': np.nan, 'Best_Threshold': np.nan, 'Best_F1': np.nan})
            # 處理掉殘留的 best_model.pth, 避免下一輪混淆
            if os.path.exists('best_model.pth'):
                os.remove('best_model.pth')


    # 7. 總結所有 LOSO 結果
    print("\n" + "="*60)
    print(" LOSO 交叉驗證最終總結報告")
    print("="*60)
    
    # 儲存最終總結報告
    results_df = pd.DataFrame(loso_results)
    summary_path = os.path.join(run_dir, f"loso_summary_{timestamp}.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\n LOSO 總結報告已儲存: {summary_path}")
    
    valid_auc = results_df['AUC'].dropna()
    valid_f1 = results_df['Best_F1'].dropna()
    
    print("\n 整體平均表現:")
    if not valid_auc.empty:
        print(f" Mean AUC: {valid_auc.mean():.4f} ± {valid_auc.std():.4f}")
        print(f" Mean F1:  {valid_f1.mean():.4f} ± {valid_f1.std():.4f}")
    
    # 儲存詳細機率大表
    if all_predictions_list:
        all_details_df = pd.concat(all_predictions_list, axis=0, ignore_index=True)
        details_filename = os.path.join(run_dir, f'loso_detailed_predictions_{timestamp}.csv')
        all_details_df.to_csv(details_filename, index=False)
        print(f"詳細預測機率已存至: {details_filename}") 

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
