"""
统一TTS服务核心逻辑
"""
import os
import hashlib
import shutil
from pathlib import Path
from typing import Literal, Optional
import time
import collections
import statistics

from ..config import config as global_config
from ..engines.cosyvoice_engine import CosyVoiceEngine
from ..engines.mms_engine import MMSEngine

from typing import Literal, Optional, Dict

class UnifiedTTSService:
    """自动路由: 英文 -> CosyVoice, 马来文 -> MMS"""
    
    def __init__(self, config=None):
        self.config = config or global_config
        self._cosyvoice = None
        self._mms = None
        self.latencies = {
            'synthesize_s': collections.deque(maxlen=100),
            'stream_ttfb_s': collections.deque(maxlen=100)
        }
        
    @property
    def cosyvoice(self):
        if self._cosyvoice is None:
            self._cosyvoice = CosyVoiceEngine(
                model_path=self.config.cosyvoice_model_path,
                device=self.config.device
            )
        return self._cosyvoice
    
    @property
    def mms(self):
        if self._mms is None:
            self._mms = MMSEngine(
                model_dir=self.config.mms_model_dir,
                device="cpu"
            )
        return self._mms
    
    def get_health(self) -> Dict:
        """检查并返回所有引擎的健康状况"""
        loaded_engines = []
        if self._cosyvoice is not None:
            loaded_engines.append("cosyvoice")
        if self._mms is not None:
            loaded_engines.append("mms")
        
        if not loaded_engines:
            # 尝试预加载，以反映真实状态
            try:
                self.cosyvoice
                self.mms
                if self._cosyvoice: loaded_engines.append("cosyvoice")
                if self._mms: loaded_engines.append("mms")
            except Exception:
                # 即使预加载失败，也返回空列表
                pass

        return {
            "status": "healthy" if loaded_engines else "unhealthy",
            "engines": sorted(list(set(loaded_engines)))
        }

    def get_metrics(self) -> Dict:
        """计算并返回延迟指标"""
        latency_stats = {}
        for key, deq in self.latencies.items():
            if not deq:
                latency_stats[key] = "no_data"
                continue
            
            latencies = list(deq)
            latency_stats[key] = {
                "count": len(latencies),
                "avg": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": statistics.quantiles(latencies, n=100, method='exclusive')[94] if len(latencies) > 1 else latencies[0],
                "max": max(latencies),
                "min": min(latencies),
            }
        return {"inference_latency": latency_stats}
    
    def _get_cache_path(self, text: str, language: str, voice: str) -> Path:
        key = hashlib.md5(f"{text}_{language}_{voice}".encode()).hexdigest()
        return Path(self.config.cache_dir) / f"{key}.wav"
    
    def synthesize(self, text: str, language: Literal["en", "ms"], voice: Optional[str] = None, output_path: Optional[str] = None, use_cache: bool = True) -> dict:
        start_time = time.time()
        
        if voice is None:
            voice = self.config.default_english_voice if language == "en" else self.config.default_malay_voice
        
        if use_cache:
            cache_path = self._get_cache_path(text, language, voice)
            if cache_path.exists():
                if output_path:
                    shutil.copy(cache_path, output_path)
                
                # 即使是缓存，也记录一次快速的响应时间
                self.latencies['synthesize_s'].append(time.time() - start_time)
                return {"success": True, "engine": "cache", "output_path": output_path or str(cache_path), "cached": True}
        
        if language == "en":
            self.cosyvoice.synthesize(text, voice=voice, output_path=output_path)
            res = {"success": True, "engine": "cosyvoice", "output_path": output_path, "sample_rate": self.cosyvoice.sample_rate}
        elif language == "ms":
            self.mms.synthesize(text, language="ms", output_path=output_path)
            res = {"success": True, "engine": "mms", "output_path": output_path, "sample_rate": self.mms.get_sample_rate("ms")}
        else:
            raise ValueError(f"Unsupported language: {language}")
            
        if output_path:
            cache_path = self._get_cache_path(text, language, voice)
            shutil.copy(output_path, cache_path)
        
        res["cached"] = False
        self.latencies['synthesize_s'].append(time.time() - start_time)
        return res

    def stream_synthesize(self, text: str, language: Literal["en", "ms"], voice: Optional[str] = None):
        """流式合成主入口"""
        start_time = time.time()
        first_chunk = True

        if voice is None:
            voice = self.config.default_english_voice if language == "en" else self.config.default_malay_voice
        
        if language == "en":
            # CosyVoice 原生支持流式
            for chunk in self.cosyvoice.synthesize_stream(text, voice):
                if first_chunk:
                    self.latencies['stream_ttfb_s'].append(time.time() - start_time)
                    first_chunk = False
                yield chunk
        elif language == "ms":
            # MMS 暂不支持流式，我们一次性生成并模拟流式返回以保持 API 兼容
            audio = self.mms.synthesize(text, language="ms")
            self.latencies['stream_ttfb_s'].append(time.time() - start_time) # 记录一次性生成的时间作为TTFB
            yield audio.tobytes()
        else:
            raise ValueError(f"Unsupported language: {language}")

_service = None
def get_service():
    global _service
    if _service is None:
        _service = UnifiedTTSService()
    return _service
