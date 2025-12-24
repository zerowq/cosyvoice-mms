# 使用支持 MPS 的基础镜像是不现实的（MPS 仅限宿主机），生产环境通常是 NVIDIA GPU
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELSCOPE_CACHE=/app/models \
    MODELSCOPE_OFFLINE=1 \
    MODELSCOPE_ENVIRONMENT=local \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/app:/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 提速安装
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 复制依赖定义
COPY pyproject.toml .

# 离线环境构建：先安装依赖 (构建阶段允许联网)
RUN uv pip install --system -r pyproject.toml

# 复制源码和工具脚本
COPY . .

# 【关键步骤】在构建镜像时预下载模型 (此时 CI 环境应有网络)
# 注意：构建完成后，models 目录将被固化在镜像中
RUN unset MODELSCOPE_OFFLINE && \
    unset HF_HUB_OFFLINE && \
    python3 scripts/download_models.py && \
    python3 scripts/create_dummy_voice.py

# 暴露端口
EXPOSE 8000

# 启动脚本
CMD ["python3", "-m", "src.main"]
