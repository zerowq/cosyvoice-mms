# Docker 部署指南

## 前置要求

1. 安装 Docker
2. 安装 NVIDIA Container Toolkit（用于 GPU 支持）

```bash
# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 构建镜像

```bash
# 在项目根目录执行
docker build -t cosyvoice-mms:latest .
```

构建时间：约 10-20 分钟（取决于网络速度）

## 运行容器

### 基本运行（使用所有 GPU）

```bash
docker run -d \
  --name cosyvoice-mms \
  --gpus all \
  -p 8080:8080 \
  cosyvoice-mms:latest
```

### 指定 GPU

```bash
# 只使用 GPU 0
docker run -d \
  --name cosyvoice-mms \
  --gpus '"device=0"' \
  -p 8080:8080 \
  cosyvoice-mms:latest

# 使用 GPU 0 和 1
docker run -d \
  --name cosyvoice-mms \
  --gpus '"device=0,1"' \
  -p 8080:8080 \
  cosyvoice-mms:latest
```

### 挂载外部模型目录（可选）

```bash
docker run -d \
  --name cosyvoice-mms \
  --gpus all \
  -p 8080:8080 \
  -v /path/to/models:/app/models \
  cosyvoice-mms:latest
```

## 验证运行

```bash
# 查看日志
docker logs -f cosyvoice-mms

# 检查健康状态
curl http://localhost:8080/api/health

# 测试页面
# 浏览器访问: http://localhost:8080/static/test_stream.html
```

## 常用命令

```bash
# 停止容器
docker stop cosyvoice-mms

# 启动容器
docker start cosyvoice-mms

# 重启容器
docker restart cosyvoice-mms

# 删除容器
docker rm -f cosyvoice-mms

# 查看容器资源使用
docker stats cosyvoice-mms

# 进入容器
docker exec -it cosyvoice-mms bash
```

## 镜像说明

- **基础镜像**: nvidia/cuda:12.3.0-cudnn9-runtime-ubuntu22.04
- **CUDA 版本**: 12.3
- **cuDNN 版本**: 9
- **Python 版本**: 3.10
- **包含组件**:
  - CosyVoice 2.0/3.0
  - MMS-TTS
  - Kokoro-82M (支持 GPU 加速)
  - 所有预下载的模型

## 故障排查

### GPU 不可用

```bash
# 检查 NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# 如果失败，重启 Docker
sudo systemctl restart docker
```

### 端口被占用

```bash
# 使用其他端口
docker run -d --name cosyvoice-mms --gpus all -p 8888:8080 cosyvoice-mms:latest
```

### 查看详细日志

```bash
docker logs --tail 100 cosyvoice-mms
```
