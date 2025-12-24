#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本 - 支持国内镜像源及本地保存
"""
import os
import sys
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()
MODELS_DIR = ROOT_DIR / "models"

def setup_mirror():
    """设置 Hugging Face 镜像源以提高下载成功率"""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("🌐 Using Hugging Face mirror: https://hf-mirror.com")

def download_cosyvoice():
    """下载 CosyVoice 2.0 模型 (从 ModelScope 下载，国内速度快)"""
    print("\n📥 [1/2] Downloading CosyVoice 2.0...")
    try:
        from modelscope import snapshot_download
        path = MODELS_DIR / "CosyVoice2-0.5B"
        snapshot_download(
            'iic/CosyVoice2-0.5B',
            local_dir=str(path)
        )
        print(f"✅ CosyVoice 2.0 downloaded to {path}")
    except Exception as e:
        print(f"❌ Error downloading CosyVoice: {e}")

def download_mms():
    """下载 MMS-TTS 模型"""
    print("\n📥 [2/2] Downloading MMS-TTS models...")
    setup_mirror()
    try:
        from transformers import VitsModel, AutoTokenizer
        
        languages = {
            "mms-tts-eng": "facebook/mms-tts-eng",
            "mms-tts-zlm": "facebook/mms-tts-zlm"
        }
        
        for name, hf_path in languages.items():
            local_path = MODELS_DIR / name
            if local_path.exists():
                print(f"  ⏩ {name} already exists, skipping...")
                continue
                
            print(f"  Downloading {name} from {hf_path}...")
            # 显式下载到本地
            model = VitsModel.from_pretrained(hf_path)
            tokenizer = AutoTokenizer.from_pretrained(hf_path)
            
            # 保存到指定的 models 目录
            model.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)
            print(f"  ✅ Saved {name} to {local_path}")
            
    except Exception as e:
        print(f"❌ Error downloading MMS models: {e}")
        print("\n💡 Manual Download Option:")
        print("If the script fails, please manually download the files from:")
        print("- https://hf-mirror.com/facebook/mms-tts-zlm/tree/main")
        print("- https://hf-mirror.com/facebook/mms-tts-eng/tree/main")
        print("And place them in: models/mms-tts-zlm/ and models/mms-tts-eng/")

def download_wetext():
    """下载 WeText 前端资源及 ModelScope 元数据"""
    print("\n📥 [3/3] Downloading WeText resources...")
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        # 显式下载到 MODELS_DIR，不再使用 local_dir 干扰元数据
        snapshot_download('pengzhendong/wetext', cache_dir=str(MODELS_DIR), local_files_only=False)
        print(f"✅ WeText resources synced to {MODELS_DIR}")
    except Exception as e:
        print(f"❌ Error downloading WeText: {e}")

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    download_cosyvoice()
    download_mms()
    download_wetext()
    print("\n🎉 Model preparation finished.")
