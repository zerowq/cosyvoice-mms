#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# 添加 src 到路径
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# 设置离线缓存
os.environ["MODELSCOPE_CACHE"] = str(ROOT_DIR / "models")

from src.engines.cosyvoice_engine import CosyVoiceEngine
from src.config import config

def list_spks():
    print(f"Loading model from {config.cosyvoice_model_path}...")
    try:
        engine = CosyVoiceEngine(config.cosyvoice_model_path, device="cpu") # 用CPU快速加载查看及避免显存问题
        voices = engine.list_voices()
        print("\n📢 Available Voices:")
        for v in voices:
            print(f" - {v}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_spks()
