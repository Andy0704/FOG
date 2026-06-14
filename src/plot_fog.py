from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]

print("📊 正在讀取並繪製步態資料，請稍候...")

# 1. 讀取資料
file_path = ROOT_DIR / "data" / "dataset" / "S01R01.txt"
df = pd.read_csv(file_path, sep=" ", header=None)

# 2. 提取需要的欄位
time_ms = df[0]       # 時間軸
ankle_x = df[1]       # 腳踝 X 軸
ankle_y = df[2]       # 腳踝 Y 軸
ankle_z = df[3]       # 腳踝 Z 軸
labels = df[10]       # 標註 (1=正常, 2=凍結)

# 3. 建立畫布
plt.figure(figsize=(15, 5))

# 4. 畫出腳踝的 XYZ 加速度波形
# 為了讓圖表不那麼亂，我們用較細的線條 (alpha=0.8)
plt.plot(time_ms, ankle_x, label='Ankle X', alpha=0.8, color='#1f77b4', linewidth=1)
plt.plot(time_ms, ankle_y, label='Ankle Y', alpha=0.8, color='#2ca02c', linewidth=1)
plt.plot(time_ms, ankle_z, label='Ankle Z', alpha=0.8, color='#ff7f0e', linewidth=1)

# 5. 💡 靈魂步驟：把「凍結 (FoG)」的區間塗成紅色！
# 只要 Label == 2，我們就在背景畫上紅色半透明區塊
plt.fill_between(time_ms, -3000, 3000, where=(labels == 2), 
                 color='red', alpha=0.3, label='FoG Event (Freezing)')

# 6. 圖表美化 (這會讓圖表看起來很專業)
plt.title("Parkinson's Disease: Ankle IMU Signal vs Freezing of Gait", fontsize=16, fontweight='bold')
plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("Acceleration (mg)", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

# 7. 儲存圖片並顯示
plt.tight_layout()
out_path = ROOT_DIR / "fog_waveform_S01R01.png"
plt.savefig(out_path, dpi=300) # 存成高畫質圖片供計畫書使用
print(f"✅ 繪圖完成！圖片已儲存為 {out_path}")
plt.show()