#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 Fun-CosyVoice 3.0 模型
"""
import os
import sys
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()
MODELS_DIR = ROOT_DIR / "models"

def download_cosyvoice3():
    """下载 Fun-CosyVoice 3.0 模型"""
    print("=" * 60)
    print("📥 Downloading Fun-CosyVoice 3.0 (Latest Version)")
    print("=" * 60)

    target_path = MODELS_DIR / "Fun-CosyVoice3-0.5B"

    if target_path.exists() and len(list(target_path.glob("*"))) > 0:
        print(f"✅ Fun-CosyVoice 3.0 already exists at {target_path}")
        print("   Delete it first if you want to re-download.")
        return True

    try:
        from modelscope import snapshot_download

        print(f"\n📦 Model: FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
        print(f"📁 Target: {target_path}")
        print(f"\n⏳ Downloading... This may take 10-20 minutes.\n")

        snapshot_download(
            'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
            local_dir=str(target_path)
        )

        print(f"\n✅ Successfully downloaded Fun-CosyVoice 3.0!")
        print(f"   Location: {target_path}")
        print(f"\n🎉 Model is ready to use!")
        return True

    except Exception as e:
        print(f"\n❌ Error downloading Fun-CosyVoice 3.0: {e}")
        import traceback
        traceback.print_exc()

        print("\n💡 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Ensure you have enough disk space (~2-3GB)")
        print("   3. Try running: pip install modelscope")
        return False

if __name__ == "__main__":
    success = download_cosyvoice3()
    sys.exit(0 if success else 1)
