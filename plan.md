```markdown
# TTS 本地部署方案：CosyVoice 2.0 + MMS-TTS

> **版本**: 1.0.0  
> **更新日期**: 2025-06-23  
> **适用场景**: 英文 + 马来文语音合成，100%本地部署，完全免费

---

## 目录

- [一、方案概述](#一方案概述)
- [二、技术选型](#二技术选型)
- [三、硬件要求](#三硬件要求)
- [四、环境安装](#四环境安装)
- [五、模型下载](#五模型下载)
- [六、代码实现](#六代码实现)
- [七、API服务](#七api服务)
- [八、Docker部署](#八docker部署)
- [九、测试验证](#九测试验证)
- [十、常见问题](#十常见问题)

---

## 一、方案概述

### 1.1 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      TTS 统一服务架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      ┌─────────────┐                            │
│                      │   TTS请求    │                            │
│                      └──────┬──────┘                            │
│                             │                                   │
│                             ▼                                   │
│                      ┌─────────────┐                            │
│                      │  语言路由    │                            │
│                      └──────┬──────┘                            │
│                             │                                   │
│              ┌──────────────┴──────────────┐                    │
│              │                             │                    │
│              ▼                             ▼                    │
│       ┌─────────────┐               ┌─────────────┐             │
│       │   英文       │               │   马来文     │             │
│       └──────┬──────┘               └──────┬──────┘             │
│              │                             │                    │
│              ▼                             ▼                    │
│       ┌─────────────┐               ┌─────────────┐             │
│       │ CosyVoice   │               │  MMS-TTS    │             │
│       │    2.0      │               │   (Meta)    │             │
│       └──────┬──────┘               └──────┬──────┘             │
│              │                             │                    │
│              └──────────────┬──────────────┘                    │
│                             │                                   │
│                             ▼                                   │
│                      ┌─────────────┐                            │
│                      │  音频输出    │                            │
│                      └─────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 方案优势

| 特性 | 说明 |
|------|------|
| **完全免费** | 无API费用，无订阅费用 |
| **本地部署** | 100%离线运行，无网络依赖 |
| **英文高质量** | CosyVoice 2.0 提供顶级音质 |
| **马来文原生支持** | MMS-TTS 原生支持马来语 |
| **生产可用** | 稳定可靠，支持高并发 |

### 1.3 语言支持

| 语言 | 引擎 | 音质 | 声音数量 |
|------|------|------|---------|
| 英文 (English) | CosyVoice 2.0 | ⭐⭐⭐⭐⭐ | 多个预设 + 语音克隆 |
| 马来文 (Bahasa Melayu) | MMS-TTS | ⭐⭐⭐ | 1个默认声音 |

---

## 二、技术选型

### 2.1 CosyVoice 2.0 (英文)

- **开发者**: 阿里巴巴 FunAudioLLM
- **GitHub**: https://github.com/FunAudioLLM/CosyVoice
- **特点**:
  - 高质量神经网络语音合成
  - 支持多种预设声音
  - 支持零样本语音克隆
  - 支持情感/语调控制
  - 支持流式输出

**官方支持语言**:
- ✅ 中文 (普通话)
- ✅ 英文
- ✅ 日文
- ✅ 韩文
- ✅ 粤语
- ❌ 马来文 (不支持)

### 2.2 MMS-TTS (马来文)

- **开发者**: Meta (Facebook AI)
- **HuggingFace**: https://huggingface.co/facebook/mms-tts-zsm
- **特点**:
  - 支持1100+种语言
  - 原生支持马来语 (zsm)
  - 模型小巧，CPU可运行
  - 推理速度快

**马来语模型**:
- `facebook/mms-tts-zsm` - 标准马来语 (推荐)
- `facebook/mms-tts-zlm` - 马来语变体
- `facebook/mms-tts-ind` - 印尼语 (相近语言)

---

## 三、硬件要求

### 3.1 最低配置

| 组件 | 要求 |
|------|------|
| GPU | 6GB 显存 (RTX 2060/3060) |
| RAM | 16GB |
| 磁盘 | 30GB 可用空间 |
| CPU | 4核以上 |

### 3.2 推荐配置

| 组件 | 要求 |
|------|------|
| GPU | 8GB+ 显存 (RTX 3080/4080/4090) |
| RAM | 32GB |
| 磁盘 | SSD 50GB |
| CPU | 8核以上 |

### 3.3 显存分配

```
CosyVoice 2.0: ~4-6GB (GPU)
MMS-TTS:       ~500MB (CPU，不占显存)
系统预留:       ~1-2GB
```

---

## 四、环境安装

### 4.1 创建Conda环境

```bash
# 创建Python 3.10环境
conda create -n tts_service python=3.10 -y
conda activate tts_service
```

### 4.2 安装PyTorch

```bash
# CUDA 11.8 版本
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 版本
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU 版本 (不推荐，速度慢)
# pip install torch torchaudio
```

### 4.3 安装依赖

```bash
# 基础依赖
pip install transformers scipy numpy fastapi uvicorn pydantic modelscope

# CosyVoice 特殊依赖
conda install -c conda-forge pynini=2.1.5 -y
```

### 4.4 克隆CosyVoice

```bash
# 克隆仓库
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

# 安装CosyVoice依赖
cd CosyVoice
pip install -r requirements.txt
cd ..
```

### 4.5 完整requirements.txt

```txt
# requirements.txt
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.35.0
scipy>=1.11.0
numpy>=1.24.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
modelscope>=1.10.0
```

---

## 五、模型下载

### 5.1 下载脚本

创建 `download_models.py`:

```python
#!/usr/bin/env python3
"""
模型下载脚本
下载 CosyVoice 2.0 和 MMS-TTS 模型
"""

import os


def download_cosyvoice():
    """下载 CosyVoice 2.0 模型"""
    print("=" * 50)
    print("📥 Downloading CosyVoice 2.0...")
    print("=" * 50)
    
    from modelscope import snapshot_download
    
    snapshot_download(
        'iic/CosyVoice2-0.5B',
        local_dir='models/CosyVoice2-0.5B'
    )
    print("✅ CosyVoice 2.0 downloaded!")


def download_mms():
    """下载 MMS-TTS 模型"""
    print("=" * 50)
    print("📥 Downloading MMS-TTS models...")
    print("=" * 50)
    
    from transformers import VitsModel, AutoTokenizer
    
    # 英文模型 (备用)
    print("  Downloading English model...")
    VitsModel.from_pretrained("facebook/mms-tts-eng")
    AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    
    # 马来文模型
    print("  Downloading Malay model...")
    VitsModel.from_pretrained("facebook/mms-tts-zsm")
    AutoTokenizer.from_pretrained("facebook/mms-tts-zsm")
    
    print("✅ MMS-TTS models downloaded!")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    download_mms()        # 先下载小的 (~100MB)
    download_cosyvoice()  # 再下载大的 (~2GB)
    
    print("\n" + "=" * 50)
    print("🎉 All models downloaded successfully!")
    print("=" * 50)
```

### 5.2 执行下载

```bash
python download_models.py
```

### 5.3 模型大小参考

| 模型 | 大小 | 下载时间 (100Mbps) |
|------|------|-------------------|
| CosyVoice2-0.5B | ~2GB | ~3分钟 |
| MMS-TTS-eng | ~100MB | ~10秒 |
| MMS-TTS-zsm | ~100MB | ~10秒 |

---

## 六、代码实现

### 6.1 项目结构

```
tts_service/
├── requirements.txt          # Python依赖
├── download_models.py        # 模型下载脚本
├── config.py                 # 配置文件
├── models/                   # 模型目录
│   ├── CosyVoice2-0.5B/     # CosyVoice模型
│   └── __init__.py
├── engines/                  # TTS引擎封装
│   ├── __init__.py
│   ├── cosyvoice_engine.py  # CosyVoice封装
│   └── mms_engine.py        # MMS封装
├── service.py               # 统一服务入口
├── api_server.py            # FastAPI服务
├── test.py                  # 测试脚本
├── CosyVoice/               # CosyVoice源码 (git clone)
├── tts_cache/               # 音频缓存目录
├── output/                  # 输出目录
└── static/                  # 静态文件目录
    └── audio/               # API生成的音频
```

### 6.2 配置文件 (config.py)

```python
# config.py
"""
TTS服务配置
"""

import os
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSConfig:
    """TTS配置类"""
    
    # 模型路径
    cosyvoice_model_path: str = "models/CosyVoice2-0.5B"
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 缓存配置
    cache_enabled: bool = True
    cache_dir: str = "./tts_cache"
    
    # 默认声音
    default_english_voice: str = "英文女"
    default_malay_voice: str = "default"
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    def __post_init__(self):
        # 确保目录存在
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


# 全局配置实例
config = TTSConfig()
```

### 6.3 CosyVoice引擎 (engines/cosyvoice_engine.py)

```python
# engines/cosyvoice_engine.py
"""
CosyVoice 2.0 TTS 引擎封装
用于: 英文语音合成 (高质量)
"""

import os
import sys
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Generator

# 添加CosyVoice路径
COSYVOICE_PATH = Path(__file__).parent.parent / "CosyVoice"
sys.path.insert(0, str(COSYVOICE_PATH))
sys.path.insert(0, str(COSYVOICE_PATH / "third_party" / "Matcha-TTS"))


class CosyVoiceEngine:
    """
    CosyVoice 2.0 英文TTS引擎
    
    Features:
    - 高质量英文语音合成
    - 多种预设声音
    - 零样本语音克隆
    - 流式输出支持
    """
    
    def __init__(
        self, 
        model_path: str = "models/CosyVoice2-0.5B",
        device: str = "cuda"
    ):
        """
        初始化CosyVoice引擎
        
        Args:
            model_path: 模型路径
            device: 运行设备 ("cuda" 或 "cpu")
        """
        self.model_path = model_path
        self.device = device
        self._model = None
        self._loaded = False
        
    def _load_model(self):
        """延迟加载模型"""
        if not self._loaded:
            from cosyvoice.cli.cosyvoice import CosyVoice2
            
            print(f"🔄 Loading CosyVoice 2.0 on {self.device}...")
            self._model = CosyVoice2(
                self.model_path,
                load_jit=True,
                load_trt=False
            )
            self._loaded = True
            print("✅ CosyVoice 2.0 loaded!")
            
        return self._model
    
    @property
    def model(self):
        """获取模型实例"""
        return self._load_model()
    
    @property
    def sample_rate(self) -> int:
        """获取采样率"""
        return self.model.sample_rate
    
    def list_voices(self) -> list:
        """
        列出可用的预设声音
        
        Returns:
            声音列表，如 ["中文女", "中文男", "英文女", "英文男", ...]
        """
        return self.model.list_available_spks()
    
    def synthesize(
        self,
        text: str,
        voice: str = "英文女",
        output_path: Optional[str] = None,
        stream: bool = False
    ) -> torch.Tensor:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            voice: 预设声音名称
            output_path: 输出文件路径 (可选)
            stream: 是否使用流式合成
            
        Returns:
            音频张量 [1, samples]
        """
        audio_chunks = []
        
        for result in self.model.inference_sft(text, voice, stream=stream):
            audio_chunks.append(result['tts_speech'])
        
        # 合并音频块
        if len(audio_chunks) > 1:
            audio = torch.cat(audio_chunks, dim=1)
        else:
            audio = audio_chunks[0]
        
        # 保存文件
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torchaudio.save(output_path, audio, self.sample_rate)
        
        return audio
    
    def synthesize_stream(
        self,
        text: str,
        voice: str = "英文女"
    ) -> Generator[torch.Tensor, None, None]:
        """
        流式合成语音
        
        Args:
            text: 要合成的文本
            voice: 预设声音名称
            
        Yields:
            音频块张量
        """
        for result in self.model.inference_sft(text, voice, stream=True):
            yield result['tts_speech']
    
    def clone_voice(
        self,
        text: str,
        reference_audio: str,
        reference_text: str,
        output_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        零样本语音克隆
        
        Args:
            text: 要合成的目标文本
            reference_audio: 参考音频文件路径
            reference_text: 参考音频的文字内容
            output_path: 输出文件路径 (可选)
            
        Returns:
            音频张量
        """
        from cosyvoice.utils.file_utils import load_wav
        
        # 加载参考音频
        prompt_speech = load_wav(reference_audio, 16000)
        
        audio_chunks = []
        for result in self.model.inference_zero_shot(
            text, reference_text, prompt_speech, stream=False
        ):
            audio_chunks.append(result['tts_speech'])
        
        audio = torch.cat(audio_chunks, dim=1) if len(audio_chunks) > 1 else audio_chunks[0]
        
        if output_path:
            torchaudio.save(output_path, audio, self.sample_rate)
        
        return audio
    
    def cross_lingual(
        self,
        text: str,
        reference_audio: str,
        output_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        跨语言语音合成 (用参考音色说其他语言)
        
        Args:
            text: 目标语言文本
            reference_audio: 参考音频文件路径
            output_path: 输出文件路径 (可选)
            
        Returns:
            音频张量
        """
        from cosyvoice.utils.file_utils import load_wav
        
        prompt_speech = load_wav(reference_audio, 16000)
        
        audio_chunks = []
        for result in self.model.inference_cross_lingual(
            text, prompt_speech, stream=False
        ):
            audio_chunks.append(result['tts_speech'])
        
        audio = torch.cat(audio_chunks, dim=1) if len(audio_chunks) > 1 else audio_chunks[0]
        
        if output_path:
            torchaudio.save(output_path, audio, self.sample_rate)
        
        return audio
```

### 6.4 MMS-TTS引擎 (engines/mms_engine.py)

```python
# engines/mms_engine.py
"""
Meta MMS-TTS 引擎封装
用于: 马来文语音合成 (原生支持)
"""

import os
import torch
import numpy as np
import scipy.io.wavfile as wav
from typing import Optional, Dict
from transformers import VitsModel, AutoTokenizer


class MMSEngine:
    """
    Meta MMS-TTS 马来文引擎
    
    Features:
    - 原生支持马来语
    - 模型小巧
    - CPU可运行
    - 支持1100+语言
    """
    
    # 支持的语言和对应模型
    LANGUAGE_MODELS = {
        "en": "facebook/mms-tts-eng",      # 英文
        "ms": "facebook/mms-tts-zsm",      # 标准马来语
        "id": "facebook/mms-tts-ind",      # 印尼语
        "zh": "facebook/mms-tts-cmn",      # 中文
    }
    
    def __init__(self, device: str = "cpu"):
        """
        初始化MMS引擎
        
        Args:
            device: 运行设备 (推荐 "cpu"，模型小不需要GPU)
        """
        self.device = device
        self._models: Dict[str, VitsModel] = {}
        self._tokenizers: Dict[str, AutoTokenizer] = {}
    
    def _load_model(self, language: str):
        """
        延迟加载指定语言的模型
        
        Args:
            language: 语言代码 ("ms", "en", etc.)
        """
        if language not in self._models:
            model_name = self.LANGUAGE_MODELS.get(language)
            if not model_name:
                available = list(self.LANGUAGE_MODELS.keys())
                raise ValueError(
                    f"Unsupported language: {language}. "
                    f"Available: {available}"
                )
            
            print(f"🔄 Loading MMS-TTS for {language}...")
            self._models[language] = VitsModel.from_pretrained(model_name).to(self.device)
            self._tokenizers[language] = AutoTokenizer.from_pretrained(model_name)
            print(f"✅ MMS-TTS ({language}) loaded!")
    
    def get_sample_rate(self, language: str = "ms") -> int:
        """
        获取指定语言模型的采样率
        
        Args:
            language: 语言代码
            
        Returns:
            采样率 (Hz)
        """
        self._load_model(language)
        return self._models[language].config.sampling_rate
    
    def list_languages(self) -> list:
        """
        列出支持的语言
        
        Returns:
            语言代码列表
        """
        return list(self.LANGUAGE_MODELS.keys())
    
    def synthesize(
        self,
        text: str,
        language: str = "ms",
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            language: 语言代码 ("ms" = 马来文, "en" = 英文)
            output_path: 输出文件路径 (可选)
            
        Returns:
            音频波形 (numpy array)
        """
        self._load_model(language)
        
        model = self._models[language]
        tokenizer = self._tokenizers[language]
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # 转换为numpy
        waveform = output.squeeze().cpu().numpy()
        
        # 保存文件
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sample_rate = model.config.sampling_rate
            wav.write(output_path, rate=sample_rate, data=waveform)
        
        return waveform
```

### 6.5 统一服务 (service.py)

```python
# service.py
"""
统一TTS服务
自动路由: 英文->CosyVoice, 马来文->MMS
"""

import os
import hashlib
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass

import torch
import torchaudio
import numpy as np
import scipy.io.wavfile as wav


@dataclass
class TTSConfig:
    """TTS配置"""
    cosyvoice_model_path: str = "models/CosyVoice2-0.5B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_enabled: bool = True
    cache_dir: str = "./tts_cache"
    default_english_voice: str = "英文女"
    default_malay_voice: str = "default"


class UnifiedTTSService:
    """
    统一TTS服务
    
    自动路由:
    - 英文 -> CosyVoice 2.0 (高质量)
    - 马来文 -> MMS-TTS (原生支持)
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        初始化服务
        
        Args:
            config: TTS配置，None则使用默认配置
        """
        self.config = config or TTSConfig()
        
        # 延迟加载引擎
        self._cosyvoice = None
        self._mms = None
        
        # 缓存目录
        if self.config.cache_enabled:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def cosyvoice(self):
        """获取CosyVoice引擎 (延迟加载)"""
        if self._cosyvoice is None:
            from engines.cosyvoice_engine import CosyVoiceEngine
            self._cosyvoice = CosyVoiceEngine(
                model_path=self.config.cosyvoice_model_path,
                device=self.config.device
            )
        return self._cosyvoice
    
    @property
    def mms(self):
        """获取MMS引擎 (延迟加载)"""
        if self._mms is None:
            from engines.mms_engine import MMSEngine
            # MMS用CPU，省GPU显存给CosyVoice
            self._mms = MMSEngine(device="cpu")
        return self._mms
    
    def _get_cache_path(self, text: str, language: str, voice: str) -> Path:
        """生成缓存文件路径"""
        key = hashlib.md5(f"{text}_{language}_{voice}".encode()).hexdigest()
        return Path(self.config.cache_dir) / f"{key}.wav"
    
    def synthesize(
        self,
        text: str,
        language: Literal["en", "ms"],
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
        use_cache: bool = True
    ) -> dict:
        """
        合成语音 (主入口)
        
        Args:
            text: 要合成的文本
            language: "en" (英文) 或 "ms" (马来文)
            voice: 声音选择 (仅英文CosyVoice有效)
            output_path: 输出文件路径
            use_cache: 是否使用缓存
            
        Returns:
            {
                "success": bool,
                "engine": str,      # "cosyvoice" 或 "mms"
                "output_path": str,
                "sample_rate": int,
                "cached": bool
            }
        """
        # 默认声音
        if voice is None:
            voice = (
                self.config.default_english_voice 
                if language == "en" 
                else self.config.default_malay_voice
            )
        
        # 检查缓存
        if use_cache and self.config.cache_enabled:
            cache_path = self._get_cache_path(text, language, voice)
            if cache_path.exists():
                if output_path:
                    import shutil
                    shutil.copy(cache_path, output_path)
                return {
                    "success": True,
                    "engine": "cache",
                    "output_path": output_path or str(cache_path),
                    "cached": True
                }
        
        # 路由到对应引擎
        if language == "en":
            result = self._synthesize_english(text, voice, output_path)
        elif language == "ms":
            result = self._synthesize_malay(text, output_path)
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        # 保存到缓存
        if self.config.cache_enabled and output_path:
            cache_path = self._get_cache_path(text, language, voice)
            import shutil
            shutil.copy(output_path, cache_path)
        
        result["cached"] = False
        return result
    
    def _synthesize_english(self, text: str, voice: str, output_path: str) -> dict:
        """英文合成 (CosyVoice)"""
        self.cosyvoice.synthesize(text, voice=voice, output_path=output_path)
        return {
            "success": True,
            "engine": "cosyvoice",
            "output_path": output_path,
            "sample_rate": self.cosyvoice.sample_rate
        }
    
    def _synthesize_malay(self, text: str, output_path: str) -> dict:
        """马来文合成 (MMS)"""
        self.mms.synthesize(text, language="ms", output_path=output_path)
        return {
            "success": True,
            "engine": "mms",
            "output_path": output_path,
            "sample_rate": self.mms.get_sample_rate("ms")
        }
    
    def synthesize_batch(
        self,
        items: list,
        output_dir: str
    ) -> list:
        """
        批量合成
        
        Args:
            items: [{"text": "...", "language": "en", "filename": "001.wav"}, ...]
            output_dir: 输出目录
            
        Returns:
            结果列表
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []
        
        for i, item in enumerate(items):
            print(f"[{i+1}/{len(items)}] {item['text'][:30]}...")
            
            output_path = os.path.join(output_dir, item["filename"])
            result = self.synthesize(
                text=item["text"],
                language=item["language"],
                voice=item.get("voice"),
                output_path=output_path
            )
            results.append({**item, **result})
        
        return results


# ==================== 便捷函数 ====================

_service: Optional[UnifiedTTSService] = None


def get_service() -> UnifiedTTSService:
    """获取全局服务实例 (单例)"""
    global _service
    if _service is None:
        _service = UnifiedTTSService()
    return _service


def speak(
    text: str, 
    language: str = "en", 
    output_path: Optional[str] = None
) -> dict:
    """
    快速语音合成
    
    Examples:
        speak("Hello world", "en", "hello.wav")
        speak("Selamat pagi", "ms", "pagi.wav")
    """
    return get_service().synthesize(text, language, output_path=output_path)
```

---

## 七、API服务

### 7.1 FastAPI服务 (api_server.py)

```python
# api_server.py
"""
TTS REST API 服务
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import uuid
import os

from service import UnifiedTTSService, TTSConfig


# ==================== 应用初始化 ====================

app = FastAPI(
    title="TTS Service API",
    description="英文 (CosyVoice) + 马来文 (MMS) TTS服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# TTS服务
tts_service = UnifiedTTSService(TTSConfig())


# ==================== 数据模型 ====================

class TTSRequest(BaseModel):
    """TTS请求"""
    text: str
    language: Literal["en", "ms"] = "en"
    voice: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "language": "en",
                "voice": "英文女"
            }
        }


class TTSResponse(BaseModel):
    """TTS响应"""
    success: bool
    engine: str
    audio_url: str
    cached: bool


class BatchItem(BaseModel):
    """批量合成项"""
    text: str
    language: Literal["en", "ms"]
    filename: str
    voice: Optional[str] = None


class BatchTTSRequest(BaseModel):
    """批量TTS请求"""
    items: list[BatchItem]


# ==================== API端点 ====================

@app.get("/")
async def root():
    """API信息"""
    return {
        "service": "TTS Service",
        "version": "1.0.0",
        "engines": {
            "english": "CosyVoice 2.0",
            "malay": "MMS-TTS (Meta)"
        },
        "endpoints": {
            "synthesize": "POST /api/tts",
            "batch": "POST /api/tts/batch",
            "voices": "GET /api/voices",
            "health": "GET /health"
        }
    }


@app.get("/api/voices")
async def list_voices():
    """列出可用声音"""
    return {
        "english": {
            "engine": "CosyVoice 2.0",
            "voices": tts_service.cosyvoice.list_voices()
        },
        "malay": {
            "engine": "MMS-TTS",
            "voices": ["default"]
        }
    }


@app.post("/api/tts", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    """
    合成语音
    
    - **text**: 要转换的文本
    - **language**: "en" (英文) 或 "ms" (马来文)
    - **voice**: 声音选择 (仅英文有效)
    """
    try:
        filename = f"{uuid.uuid4()}.wav"
        output_path = f"static/audio/{filename}"
        
        result = tts_service.synthesize(
            text=request.text,
            language=request.language,
            voice=request.voice,
            output_path=output_path
        )
        
        return TTSResponse(
            success=True,
            engine=result["engine"],
            audio_url=f"/static/audio/{filename}",
            cached=result["cached"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts/batch")
async def synthesize_batch(request: BatchTTSRequest):
    """批量合成语音"""
    try:
        batch_id = str(uuid.uuid4())[:8]
        output_dir = f"static/audio/batch_{batch_id}"
        
        items = [item.model_dump() for item in request.items]
        results = tts_service.synthesize_batch(items, output_dir)
        
        return {
            "success": True,
            "batch_id": batch_id,
            "results": [
                {
                    "text": r["text"][:50],
                    "language": r["language"],
                    "engine": r["engine"],
                    "audio_url": f"/static/audio/batch_{batch_id}/{r['filename']}"
                }
                for r in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "engines": {
            "cosyvoice": "loaded" if tts_service._cosyvoice else "standby",
            "mms": "loaded" if tts_service._mms else "standby"
        }
    }


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 7.2 启动API服务

```bash
# 开发模式
python api_server.py

# 或使用uvicorn (支持热重载)
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 7.3 API使用示例

```bash
# 英文合成
curl -X POST "http://localhost:8000/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?", "language": "en"}'

# 马来文合成
curl -X POST "http://localhost:8000/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Selamat pagi, apa khabar?", "language": "ms"}'

# 列出可用声音
curl "http://localhost:8000/api/voices"

# 健康检查
curl "http://localhost:8000/health"
```

---

## 八、Docker部署

### 8.1 Dockerfile

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 设置时区
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 系统依赖
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装pynini (CosyVoice依赖)
RUN conda install -y -c conda-forge pynini=2.1.5

# 克隆CosyVoice
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
RUN cd CosyVoice && pip install -r requirements.txt

# 预下载MMS模型 (构建时下载，加快启动)
RUN python -c "from transformers import VitsModel, AutoTokenizer; \
    VitsModel.from_pretrained('facebook/mms-tts-eng'); \
    VitsModel.from_pretrained('facebook/mms-tts-zsm')"

# 复制代码
COPY . .

# 创建目录
RUN mkdir -p static/audio tts_cache output

EXPOSE 8000

# 启动命令
CMD ["python", "api_server.py"]
```

### 8.2 docker-compose.yml

```yaml
# docker-compose.yml
version: '3.8'

services:
  tts:
    build: .
    container_name: tts-service
    ports:
      - "8000:8000"
    volumes:
      # 模型目录 (挂载本地已下载的模型)
      - ./models:/app/models
      # 缓存目录
      - ./tts_cache:/app/tts_cache
      # 输出目录
      - ./output:/app/output
      # 音频静态文件
      - ./static:/app/static
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 8.3 Docker部署命令

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## 九、测试验证

### 9.1 测试脚本 (test.py)

```python
# test.py
"""
TTS服务测试脚本
"""

import os
import time
from service import UnifiedTTSService, TTSConfig, speak

os.makedirs("output", exist_ok=True)


def test_basic():
    """基础功能测试"""
    print("=" * 60)
    print("🧪 Test 1: 基础功能测试")
    print("=" * 60)
    
    # 英文测试
    print("\n📢 英文测试 (CosyVoice)...")
    result = speak(
        "Hello, welcome to our service. How can I help you today?",
        language="en",
        output_path="output/test_english.wav"
    )
    print(f"   ✅ Engine: {result['engine']}")
    print(f"   ✅ Output: {result['output_path']}")
    
    # 马来文测试
    print("\n📢 马来文测试 (MMS)...")
    result = speak(
        "Selamat pagi, apa khabar hari ini?",
        language="ms",
        output_path="output/test_malay.wav"
    )
    print(f"   ✅ Engine: {result['engine']}")
    print(f"   ✅ Output: {result['output_path']}")


def test_voices():
    """声音测试"""
    print("\n" + "=" * 60)
    print("🧪 Test 2: 多声音测试")
    print("=" * 60)
    
    service = UnifiedTTSService()
    
    # 列出声音
    voices = service.cosyvoice.list_voices()
    print(f"\n📢 可用声音: {voices}")
    
    # 测试英文男声
    print("\n📢 英文男声测试...")
    service.synthesize(
        "This is a test with male voice.",
        language="en",
        voice="英文男",
        output_path="output/test_english_male.wav"
    )
    print("   ✅ output/test_english_male.wav")


def test_batch():
    """批量合成测试"""
    print("\n" + "=" * 60)
    print("🧪 Test 3: 批量合成测试")
    print("=" * 60)
    
    service = UnifiedTTSService()
    
    items = [
        {"text": "Welcome to our service.", "language": "en", "filename": "batch_en_01.wav"},
        {"text": "Thank you for calling.", "language": "en", "filename": "batch_en_02.wav"},
        {"text": "Selamat datang.", "language": "ms", "filename": "batch_ms_01.wav"},
        {"text": "Terima kasih.", "language": "ms", "filename": "batch_ms_02.wav"},
    ]
    
    results = service.synthesize_batch(items, "output/batch")
    
    print("\n📊 批量结果:")
    for r in results:
        print(f"   [{r['language']}] {r['engine']:10} -> {r['filename']}")


def test_cache():
    """缓存性能测试"""
    print("\n" + "=" * 60)
    print("🧪 Test 4: 缓存性能测试")
    print("=" * 60)
    
    service = UnifiedTTSService(TTSConfig(cache_enabled=True))
    text = "This is a cache test sentence for performance measurement."
    
    # 第一次 (无缓存)
    start = time.time()
    result1 = service.synthesize(text, "en", output_path="output/cache_test1.wav")
    time1 = time.time() - start
    
    # 第二次 (有缓存)
    start = time.time()
    result2 = service.synthesize(text, "en", output_path="output/cache_test2.wav")
    time2 = time.time() - start
    
    print(f"\n   首次调用: {time1:.3f}s (cached={result1['cached']})")
    print(f"   缓存调用: {time2:.3f}s (cached={result2['cached']})")
    print(f"   ⚡ 加速比: {time1/max(time2, 0.001):.1f}x")


def test_malay_sentences():
    """马来文句子测试"""
    print("\n" + "=" * 60)
    print("🧪 Test 5: 马来文句子测试")
    print("=" * 60)
    
    sentences = [
        "Selamat pagi, apa khabar?",
        "Terima kasih kerana menghubungi kami.",
        "Pesanan anda telah berjaya diproses.",
        "Sila tunggu sebentar.",
        "Selamat datang ke perkhidmatan pelanggan.",
    ]
    
    service = UnifiedTTSService()
    
    print("\n📢 生成马来文音频:")
    for i, text in enumerate(sentences):
        output = f"output/malay_{i+1:02d}.wav"
        service.synthesize(text, "ms", output_path=output)
        print(f"   ✅ [{i+1}] {text[:35]}...")


def main():
    """运行所有测试"""
    print("\n" + "🎯" * 30)
    print("         TTS 服务测试")
    print("🎯" * 30)
    
    test_basic()
    test_voices()
    test_batch()
    test_cache()
    test_malay_sentences()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成!")
    print("📂 请检查 output/ 目录中的音频文件")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

### 9.2 运行测试

```bash
python test.py
```

### 9.3 预期输出

```
🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯
         TTS 服务测试
🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯

============================================================
🧪 Test 1: 基础功能测试
============================================================

📢 英文测试 (CosyVoice)...
🔄 Loading CosyVoice 2.0 on cuda...
✅ CosyVoice 2.0 loaded!
   ✅ Engine: cosyvoice
   ✅ Output: output/test_english.wav

📢 马来文测试 (MMS)...
🔄 Loading MMS-TTS for ms...
✅ MMS-TTS (ms) loaded!
   ✅ Engine: mms
   ✅ Output: output/test_malay.wav

...

============================================================
🎉 所有测试完成!
📂 请检查 output/ 目录中的音频文件
============================================================
```

---

## 十、常见问题

### Q1: CUDA内存不足

**错误信息**: `CUDA out of memory`

**解决方案**:
```python
# 在 config.py 中设置
device = "cpu"  # 使用CPU运行

# 或减少批处理大小
# 或释放其他GPU进程
```

### Q2: CosyVoice模型加载失败

**错误信息**: `ModuleNotFoundError: No module named 'cosyvoice'`

**解决方案**:
```bash
# 确保路径正确
cd CosyVoice
pip install -r requirements.txt

# 检查路径添加
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/third_party/Matcha-TTS
```

### Q3: pynini安装失败

**解决方案**:
```bash
# 使用conda安装
conda install -c conda-forge pynini=2.1.5

# 如果还是失败，尝试
pip install pynini==2.1.5
```

### Q4: 马来文发音不准确

MMS-TTS的马来文音质可能不如商用API，这是开源模型的局限。

**建议**:
1. 检查文本是否有拼写错误
2. 避免混合英文单词
3. 如需更高音质，考虑使用Azure TTS API

### Q5: 如何添加新语言?

```python
# 在 engines/mms_engine.py 中添加
LANGUAGE_MODELS = {
    "en": "facebook/mms-tts-eng",
    "ms": "facebook/mms-tts-zsm",
    "id": "facebook/mms-tts-ind",  # 添加印尼语
    "th": "facebook/mms-tts-tha",  # 添加泰语
    # ... 更多语言
}
```

### Q6: 如何提高性能?

1. **启用缓存**: 相同文本直接返回缓存
2. **JIT编译**: CosyVoice使用 `load_jit=True`
3. **批量处理**: 使用 `synthesize_batch` 方法
4. **GPU加速**: 确保使用CUDA设备

---

## 附录

### A. 马来文常用句子示例

```python
MALAY_SAMPLES = [
    "Selamat pagi.",                           # 早上好
    "Apa khabar?",                             # 你好吗?
    "Terima kasih.",                           # 谢谢
    "Sama-sama.",                              # 不客气
    "Selamat tinggal.",                        # 再见
    "Sila tunggu sebentar.",                   # 请稍等
    "Boleh saya bantu anda?",                  # 我能帮你吗?
    "Pesanan anda telah diproses.",            # 您的订单已处理
    "Terima kasih kerana menghubungi kami.",   # 感谢联系我们
    "Selamat datang ke perkhidmatan kami.",    # 欢迎使用我们的服务
]
```

### B. 参考链接

- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [MMS-TTS HuggingFace](https://huggingface.co/facebook/mms-tts-zsm)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [PyTorch 安装](https://pytorch.org/get-started/locally/)

---

> **文档维护**: 如有问题或建议，请联系开发团队
```

---

这个MD文件可以直接复制使用。需要我调整格式或添加其他内容吗？