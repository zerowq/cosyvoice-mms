#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 Fun-CosyVoice 3.0 模型
支持 aria2c 多线程下载
"""
import os
import sys
import subprocess
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()
MODELS_DIR = ROOT_DIR / "models"

def check_aria2c():
    """检查 aria2c 是否可用"""
    try:
        result = subprocess.run(['aria2c', '--version'],
                              capture_output=True,
                              timeout=5)
        return result.returncode == 0
    except:
        return False

def download_with_aria2c():
    """使用 aria2c 多线程下载模型文件"""
    print("=" * 60)
    print("📥 Downloading Fun-CosyVoice 3.0 with aria2c")
    print("=" * 60)

    target_path = MODELS_DIR / "Fun-CosyVoice3-0.5B"
    target_path.mkdir(parents=True, exist_ok=True)

    # ModelScope 文件 URL 模式
    model_id = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    base_url = f"https://www.modelscope.cn/api/v1/models/{model_id}/repo?Revision=master&FilePath="

    # Fun-CosyVoice3-0.5B 实际文件列表
    files = [
        "campplus.onnx",
        "configuration.json",
        "cosyvoice3.yaml",
        "flow.pt",
        "flow.decoder.estimator.fp32.onnx",
        "hift.pt",
        "llm.pt",
        "llm.rl.pt",
        "speech_tokenizer_v3.onnx",
        "README.md",
        "asset/dingding.png",
        "CosyVoice-BlankEN/config.json",
        "CosyVoice-BlankEN/generation_config.json",
        "CosyVoice-BlankEN/merges.txt",
        "CosyVoice-BlankEN/model.safetensors",
        "CosyVoice-BlankEN/tokenizer_config.json",
        "CosyVoice-BlankEN/vocab.json",
    ]

    print(f"\n📁 Target directory: {target_path}\n")
    print("⏳ Downloading files with 16 connections per file...\n")

    for file in files:
        file_url = base_url + file
        output_file = target_path / file

        # 确保子目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists():
            print(f"⏩ Skipping {file} (already exists)")
            continue

        print(f"📥 Downloading {file}...")
        cmd = [
            'aria2c',
            '-x', '16',          # 16 connections
            '-s', '16',          # 16 splits
            '-k', '1M',          # chunk size
            '--file-allocation=none',
            '-d', str(output_file.parent),
            '-o', output_file.name,
            file_url
        ]

        try:
            result = subprocess.run(cmd, check=True)
            print(f"✅ Downloaded {file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download {file}: {e}")
            return False

    print(f"\n✅ All files downloaded to {target_path}")
    return True

def download_with_modelscope():
    """使用 ModelScope SDK 下载（单线程，但简单可靠）"""
    print("=" * 60)
    print("📥 Downloading Fun-CosyVoice 3.0 with ModelScope SDK")
    print("=" * 60)

    target_path = MODELS_DIR / "Fun-CosyVoice3-0.5B"

    if target_path.exists() and len(list(target_path.glob("*"))) > 0:
        print(f"✅ Fun-CosyVoice 3.0 already exists at {target_path}")
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

        print(f"\n✅ Successfully downloaded!")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download Fun-CosyVoice 3.0")
    parser.add_argument("--use-aria2c", action="store_true",
                       help="Use aria2c for multi-threaded download (faster)")
    parser.add_argument("--use-sdk", action="store_true",
                       help="Use ModelScope SDK (simpler, single-threaded)")
    args = parser.parse_args()

    target_path = MODELS_DIR / "Fun-CosyVoice3-0.5B"

    # 检查是否已存在
    if target_path.exists() and len(list(target_path.glob("*"))) > 5:
        print(f"✅ Fun-CosyVoice 3.0 already exists at {target_path}")
        print("   Delete it first if you want to re-download.")
        return True

    # 决定使用哪种方法
    if args.use_aria2c:
        if not check_aria2c():
            print("❌ aria2c not found. Install it with:")
            print("   Ubuntu/Debian: sudo apt-get install aria2")
            print("   macOS: brew install aria2")
            return False
        return download_with_aria2c()
    elif args.use_sdk:
        return download_with_modelscope()
    else:
        # 自动选择
        if check_aria2c():
            print("🚀 aria2c detected, using multi-threaded download for speed!")
            print("   (Use --use-sdk if you prefer ModelScope SDK)\n")
            return download_with_aria2c()
        else:
            print("📦 Using ModelScope SDK (aria2c not available)")
            print("   Install aria2c for faster multi-threaded download\n")
            return download_with_modelscope()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
