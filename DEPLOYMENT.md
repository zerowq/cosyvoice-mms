# CosyVoice-MMS TTS Service Deployment Manual / 部署手册

## 1. Service Overview / 服务概述

- **Service Name (服务名称)**: `cosyvoice-mms-tts`
- **Purpose (用途)**: Provides offline, high-quality Text-to-Speech (TTS). Supports English (CosyVoice 2.0) with zero-shot cloning and Malay (Meta MMS). / 提供离线、高质量的语音合成服务。支持英语（CosyVoice 2.0，带零样本克隆）和马来语（Meta MMS）。
- **Resource Requirements (资源要求)**: 
  - **Hardware (硬件)**: NVIDIA GPU (16GB+ VRAM recommended for 0.5B model). / NVIDIA GPU（建议 16GB 以上显存以支持 0.5B 模型）。
  - **Storage (存储)**: Initial image size ~15GB (includes baked-in models). / 初始镜像约 15GB（包含内置模型权重）。
- **Port (端口)**: `8000` (HTTP).
- **Offline Mode (离线模式)**: 100% Air-gapped compatible. No external internet access required during runtime. / 100% 物理隔离兼容，运行时无需任何外部网络连接。

---

## 2. Deployment Steps / 部署步骤

### 2.1 Pull Image / 拉取镜像
The image is a "Fat Image" containing all model weights. / 镜像为“胖镜像”，已内置所有模型权重。
```bash
docker pull <YOUR_ECR_REGISTRY_PATH>/cosyvoice-mms:latest
```

### 2.2 Start Container (GPU Mode) / 启动容器 (GPU 模式)
```bash
docker run -d \
  --name tts-service \
  --restart always \
  --gpus all \
  -p 8000:8000 \
  -e ENVIRONMENT=prod \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e MODELSCOPE_OFFLINE=1 \
  <YOUR_ECR_REGISTRY_PATH>/cosyvoice-mms:latest
```
*Note: The `--gpus all` parameter is mandatory to enable hardware acceleration. / 注意：必须添加 `--gpus all` 参数以启用 GPU 硬件加速。*

---

## 3. Verification / 验证部署

### 3.1 Check Startup Logs / 检查启动日志
Confirm the models are loaded on CUDA. / 确认模型已在 CUDA 上加载。
```bash
docker logs tts-service | grep "loaded on cuda"
# Expected output: ✅ CosyVoice 2.0 loaded on cuda (fp16=True)!
```

### 3.2 Health Check / 健康检查
```bash
curl http://localhost:8000/api/health
# Expected response: {"status": "healthy", "engines": ["cosyvoice", "mms"]}
```

### 3.3 Streaming Functional Test / 流式功能测试
Access the built-in test page via browser. / 通过浏览器访问内置测试页面。
- **URL**: `http://<SERVER_IP>:8000/static/test_stream.html`

---

## 4. Domain & Network / 域名配置

- **Suggested Domain (建议域名)**: `tts-service-bn.evyd.io`
- **Backend Service (后端服务)**: `<GPU_MACHINE_IP>:8000`
- **Path Forwarding (路径转发)**: `/`
- **Note**: Ensure Nginx/Ingress supports long-lived HTTP connections for streaming responses. / 确保 Nginx/Ingress 支持长连接以维持流式响应。

---

## 5. Monitoring / 监控与告警

- **Health Check Endpoint (健康检查接口)**: `/api/health`
- **Metrics Endpoint (指标接口)**: `/api/metrics` (Provides GPU/Memory/Inference Latency).
- **Critical Alert (关键告警)**: Alert if `RTF > 1.0` (Real-Time Factor). It indicates the generation speed is slower than playback, causing audio stuttering. / 如果 `RTF > 1.0`（实时率）则触发告警，这表示生成速度慢于播放速度，会导致音频卡顿。
