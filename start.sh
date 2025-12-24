#!/bin/bash
# 启动脚本

echo "🚀 Starting TTS Service..."

# 确保必要的目录存在
mkdir -p static/audio tts_cache logs output

# 启动 FastAPI 应用
# 设置 PYTHONPATH 确保可以找到 src
export PYTHONPATH=$PYTHONPATH:$(pwd)

python src/main.py
