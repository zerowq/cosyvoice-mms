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

def check_models_exist():
    """检查所有必需的模型是否已存在"""
    # 检查 CosyVoice（优先检查 3.0，回退到 2.0）
    cosyvoice_v3 = MODELS_DIR / "Fun-CosyVoice3-0.5B"
    cosyvoice_v2 = MODELS_DIR / "CosyVoice2-0.5B"
    has_cosyvoice = (cosyvoice_v3.exists() and len(list(cosyvoice_v3.glob("*"))) > 0) or \
                     (cosyvoice_v2.exists() and len(list(cosyvoice_v2.glob("*"))) > 0)

    required_models = [
        ("CosyVoice", has_cosyvoice),
        ("mms-tts-eng", (MODELS_DIR / "mms-tts-eng").exists()),
        ("mms-tts-zlm", (MODELS_DIR / "mms-tts-zlm").exists()),
        ("kokoro", (MODELS_DIR / "kokoro").exists())
    ]

    missing_models = [name for name, exists in required_models if not exists]
    return len(missing_models) == 0, missing_models

def download_cosyvoice():
    """下载 Fun-CosyVoice 3.0 模型 (最新版本，从 ModelScope 下载)"""
    print("\n📥 [1/2] Downloading Fun-CosyVoice 3.0 (latest version)...")
    try:
        from modelscope import snapshot_download
        path = MODELS_DIR / "Fun-CosyVoice3-0.5B"
        snapshot_download(
            'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
            local_dir=str(path)
        )
        print(f"✅ Fun-CosyVoice 3.0 downloaded to {path}")
    except Exception as e:
        print(f"❌ Error downloading CosyVoice: {e}")
        print("\n💡 Fallback: Trying CosyVoice 2.0...")
        try:
            path = MODELS_DIR / "CosyVoice2-0.5B"
            snapshot_download(
                'iic/CosyVoice2-0.5B',
                local_dir=str(path)
            )
            print(f"✅ CosyVoice 2.0 downloaded to {path}")
        except Exception as e2:
            print(f"❌ Error downloading CosyVoice 2.0: {e2}")

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
    print("\n📥 [3/4] Downloading WeText resources...")
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        # 显式下载到 MODELS_DIR，不再使用 local_dir 干扰元数据
        snapshot_download('pengzhendong/wetext', cache_dir=str(MODELS_DIR), local_files_only=False)
        print(f"✅ WeText resources synced to {MODELS_DIR}")
    except Exception as e:
        print(f"❌ Error downloading WeText: {e}")

def download_kokoro():
    """下载 Kokoro-82M 模型"""
    print("\n📥 [4/4] Downloading Kokoro-82M model...")
    try:
        import urllib.request

        kokoro_dir = MODELS_DIR / "kokoro"
        kokoro_dir.mkdir(parents=True, exist_ok=True)

        # Kokoro 模型文件 URL (从 GitHub releases)
        model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"

        model_path = kokoro_dir / "kokoro-v1.0.onnx"
        voices_path = kokoro_dir / "voices.json"

        if model_path.exists() and voices_path.exists():
            print(f"  ⏩ Kokoro model already exists, skipping...")
            return

        # 下载模型文件
        if not model_path.exists():
            print(f"  Downloading kokoro-v1.0.onnx...")
            urllib.request.urlretrieve(model_url, model_path)
            print(f"  ✅ Downloaded kokoro-v1.0.onnx")

        # 下载音色文件
        if not voices_path.exists():
            print(f"  Downloading voices.json...")
            urllib.request.urlretrieve(voices_url, voices_path)
            print(f"  ✅ Downloaded voices.json")

        print(f"✅ Kokoro-82M model saved to {kokoro_dir}")

    except Exception as e:
        print(f"❌ Error downloading Kokoro model: {e}")
        print("\n💡 Manual Download Option:")
        print("Download from: https://github.com/thewh1teagle/kokoro-onnx/releases")
        print("And place files in: models/kokoro/")

def main(auto_download=False):
    """
    主函数

    Args:
        auto_download: 是否自动下载缺失的模型
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_exist, missing = check_models_exist()

    if all_exist:
        print("✅ All models are ready!")
        return True

    if not auto_download:
        print(f"⚠️  Missing models: {', '.join(missing)}")
        print("Run 'python scripts/download_models.py' to download them")
        return False

    print(f"📥 Downloading missing models: {', '.join(missing)}")
    download_cosyvoice()
    download_mms()
    download_wetext()
    download_kokoro()
    print("\n🎉 Model preparation finished.")
    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download TTS models")
    parser.add_argument("--auto", action="store_true", help="Auto download missing models")
    args = parser.parse_args()

    success = main(auto_download=True)
    sys.exit(0 if success else 1)
