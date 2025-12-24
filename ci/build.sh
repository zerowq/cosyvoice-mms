#!/bin/bash
set -e

# 定制化构建脚本，模仿 chatbot-guardrails
IMAGE_NAME="cosyvoice-mms"
TAG=${CI_COMMIT_SHORT_SHA:-latest}

echo "Building Docker image: $IMAGE_NAME:$TAG"

# 在离线 CI 中，确保模型目录已经存在于 build context
if [ ! -d "models" ]; then
    echo "Warning: models directory not found. Building without models..."
fi

docker build -t $IMAGE_NAME:$TAG .

echo "Build complete."
