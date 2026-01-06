# Scripts 目录说明

本目录包含用于模型下载、环境检查、性能测试和API验证的工具脚本。

## 📥 模型下载

### download_models.py
自动下载所有必需的TTS模型文件。

**功能:**
- 下载 CosyVoice 2.0 模型 (从 ModelScope)
- 下载 MMS-TTS 英文/马来文模型 (从 HuggingFace)
- 下载 Kokoro-82M 模型 (从 GitHub Releases)
- 检查已存在的模型，跳过重复下载

**使用方法:**
```bash
# 自动下载所有缺失的模型
python scripts/download_models.py

# 或使用 --auto 参数
python scripts/download_models.py --auto
```

**说明:** 服务首次启动时会自动调用此脚本下载模型。

---

## 🔍 环境检查

### check_env.py
检查 Python 环境、PyTorch 配置和模型文件状态。

**功能:**
- 检查 Python 和 PyTorch 版本
- 检查 CUDA/MPS (Apple Silicon) 可用性
- 检查 models/ 目录和 CosyVoice 源码

**使用方法:**
```bash
python scripts/check_env.py
```

### test_startup.py
测试项目基础导入和配置加载。

**功能:**
- 验证 src.config 配置加载
- 测试引擎模块导入
- 检查 CosyVoice 目录存在性

**使用方法:**
```bash
python scripts/test_startup.py
```

---

## 🎤 音色和模型测试

### list_voices.py
列出 CosyVoice 模型中所有可用的预设音色。

**功能:**
- 加载 CosyVoice 模型
- 显示所有可用的音色名称

**使用方法:**
```bash
python scripts/list_voices.py
```

### test_kokoro.py
测试 Kokoro-82M 模型的加载速度和合成性能。

**功能:**
- 测试模型加载时间
- 测试英文语音合成
- 生成测试音频文件到 output/

**使用方法:**
```bash
python scripts/test_kokoro.py
```

### test_kokoro_malay.py
测试 Kokoro 模型的马来语合成能力。

**使用方法:**
```bash
python scripts/test_kokoro_malay.py
```

---

## 🌐 API 测试

### test_api.py
测试本地运行的 TTS 服务 API 端点。

**功能:**
- 测试 /health 健康检查
- 测试 /api/tts 英文合成 (CosyVoice)
- 测试 /api/tts 马来文合成 (MMS)

**使用方法:**
```bash
# 确保服务已启动 (默认端口 8080)
python src/main.py

# 在另一个终端运行测试
python scripts/test_api.py
```

**注意:** 默认连接 `http://localhost:8000`，如需修改请编辑脚本中的 `BASE_URL`。

### test_api_performance.py
测试已启动服务的真实响应速度（不含模型加载时间）。

**功能:**
- 健康检查验证
- 预热请求（可选）
- 测试短句、中句、长句的响应时间
- 统计平均/最快/最慢响应时间
- 支持流式 API 测试

**使用方法:**
```bash
# 基础测试（英文）
python scripts/test_api_performance.py --url http://localhost:8080 --language en

# 测试马来文
python scripts/test_api_performance.py --url http://localhost:8080 --language ms

# 测试 Kokoro 引擎
python scripts/test_api_performance.py --url http://localhost:8080 --language kokoro

# 同时测试流式 API
python scripts/test_api_performance.py --url http://localhost:8080 --language en --stream

# 跳过预热请求
python scripts/test_api_performance.py --url http://localhost:8080 --no-warmup
```

---

## 📊 性能基准测试

### benchmark_tts.py
对比 CosyVoice 2.0 和 Kokoro-82M 的性能和资源占用。

**功能:**
- 分别测试模型加载时间、预热时间、纯合成时间
- 测量 GPU 显存占用
- 对比短句、中句、长句的合成速度
- 生成音频文件到 output/benchmark/

**使用方法:**
```bash
python scripts/benchmark_tts.py
```

**输出:**
- 模型加载时间对比
- 预热时间对比
- GPU 显存占用对比
- 纯合成速度对比（不含加载时间）
- 生成的音频文件路径

**访问结果:**
```
本地: output/benchmark/
服务: http://your-domain/output/benchmark/
```

---

## 🛠️ 工具脚本

### create_dummy_voice.py
创建用于测试的虚拟音频文件。

**功能:**
- 生成简单的正弦波音频文件
- 用于测试音频处理流程

**使用方法:**
```bash
python scripts/create_dummy_voice.py
```

---

## 📝 使用建议

**首次部署流程:**
1. `python scripts/check_env.py` - 检查环境
2. `python scripts/download_models.py` - 下载模型
3. `python scripts/test_startup.py` - 验证配置
4. `python src/main.py` - 启动服务
5. `python scripts/test_api.py` - 测试 API

**性能评估流程:**
1. `python scripts/benchmark_tts.py` - 离线性能对比
2. `python scripts/test_api_performance.py` - 在线 API 性能测试

**问题排查:**
- 模型加载失败 → 运行 `check_env.py` 和 `list_voices.py`
- API 响应异常 → 运行 `test_api.py` 检查端点
- 性能问题 → 运行 `benchmark_tts.py` 和 `test_api_performance.py` 对比
