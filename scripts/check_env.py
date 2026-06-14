import sys

def check_environment():
    print("=" * 50)
    print(f"🐍 Python 版本: {sys.version.split()[0]}")
    print("=" * 50)

    packages = [
        ('PyTorch', 'torch'),
        ('NumPy', 'numpy'),
        ('SciPy', 'scipy'),
        ('Scikit-Learn', 'sklearn'),
        ('Matplotlib', 'matplotlib'),
        ('Pandas', 'pandas')
    ]

    all_passed = True

    for name, module_str in packages:
        try:
            module = __import__(module_str)
            version = getattr(module, '__version__', '未知版本')
            print(f"✅ {name:.<15} 安裝成功 (版本: {version})")
        except ImportError:
            print(f"❌ {name:.<15} 未安裝!")
            all_passed = False

    print("=" * 50)

    if all_passed:
        import torch
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        print(f"⚡ PyTorch 目前可用運算裝置: {device}")
        print("\n🎉 恭喜！所有環境裝備已就緒，隨時可以開始訓練！")
    else:
        print("\n⚠️ 仍有套件缺失，請確認剛剛的 pip install 指令是否完整跑完。")

if __name__ == "__main__":
    check_environment()
