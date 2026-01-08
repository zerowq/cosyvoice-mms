# 使用本地已有的 CUDA 12.2 镜像
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# 安装 cuDNN 9（onnxruntime-gpu 1.23.2 需要）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

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

# 先复制模型下载脚本（这些不常变）
COPY scripts/download_models.py scripts/
COPY scripts/create_dummy_voice.py scripts/
COPY CosyVoice CosyVoice/

# 【关键步骤】在构建镜像时预下载模型 (此时 CI 环境应有网络)
# 注意：构建完成后，models 目录将被固化在镜像中
RUN unset MODELSCOPE_OFFLINE && \
    unset HF_HUB_OFFLINE && \
    python3 scripts/download_models.py && \
    python3 scripts/create_dummy_voice.py

# 最后复制业务代码（这些经常变，放最后利用缓存）
COPY src src/
COPY static static/
COPY scripts scripts/
COPY README.md .

# 暴露端口
EXPOSE 8080

# 启动脚本
CMD ["python3", "-m", "src.main"]
