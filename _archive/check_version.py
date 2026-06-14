# 1. 載入套件並檢查版本
import pandas as pd
import matplotlib

print("=" * 30)
print(f"✅ Pandas 版本: {pd.__version__}")
print(f"✅ Matplotlib 版本: {matplotlib.__version__}")
print("=" * 30)

# 2. 測試讀取 Daphnet 資料集
# 請確保你已經下載了資料，並把檔案名稱換成你電腦裡實際的路徑
file_path = "C:/Project/115Daphnet_FoG/data/dataset/S01R01.txt"  # 如果檔案跟程式碼放一起，直接寫檔名即可

try:
    # 關鍵參數解析：
    # delim_whitespace=True: 告訴 pandas 資料是用「空格」隔開，而不是逗號
    # header=None: 告訴 pandas 這個檔案第一行「不是」標題，請自動補上 0, 1, 2... 的欄位編號
    df = pd.read_csv(file_path, sep=" ", header=None)
    
    print("\n🎉 成功讀取資料！資料表的形狀 (列數, 欄數) 是:", df.shape)
    print("\n👇 這是資料的前 5 筆長相：")
    print(df.head())

except FileNotFoundError:
    print(f"\n❌ 找不到檔案 '{file_path}'，請檢查路徑對不對喔！")