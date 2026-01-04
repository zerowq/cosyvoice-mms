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
from ..engines.kokoro_engine import KokoroEngine

from typing import Literal, Optional, Dict

class UnifiedTTSService:
    """自动路由: 英文 -> CosyVoice, 马来文 -> MMS"""
    
    def __init__(self, config=None):
        self.config = config or global_config
        self._cosyvoice = None
        self._mms = None
        self._kokoro = None
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
    
    @property
    def kokoro(self):
        if self._kokoro is None:
            from pathlib import Path
            root_dir = Path(__file__).parent.parent.parent.absolute()
            model_path = str(root_dir / "models" / "kokoro" / "kokoro-v1.0.onnx")
            voices_path = str(root_dir / "models" / "kokoro" / "voices.json")
            self._kokoro = KokoroEngine(model_path, voices_path)
        return self._kokoro
    
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

    def stream_synthesize(self, text: str, language: Literal["en", "ms", "kokoro", "kokoro_ms"], voice: Optional[str] = None):
        """流式合成主入口，带全引擎性能监控"""
        import time
        start_time = time.time()
        first_chunk_time = None
        total_audio_len = 0
        
        if voice is None or voice == "default":
            if language in ["kokoro", "kokoro_ms"]:
                voice = "af_sarah"
            elif language == "en":
                voice = self.config.default_english_voice
            else:
                voice = "default" # MMS 保持 default

        print(f"🚀 [TTS] Request {language.upper()} started: '{text[:30]}...' [Voice: {voice}]")

        try:
            if language == "en":
                # CosyVoice 原生支持流式
                for chunk_bytes in self.cosyvoice.synthesize_stream(text, voice):
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    
                    # CosyVoice 返回的是 float32, 22050Hz
                    # 注意：synthesize_stream 内部现在返回的是 .tobytes()
                    chunk_duration = (len(chunk_bytes) / 4) / 22050
                    total_audio_len += chunk_duration
                    yield chunk_bytes
            
            elif language in ["kokoro", "kokoro_ms"]:
                # Kokoro-82M 轻量化 TTS
                # 如果是 kokoro_ms，逻辑上依然使用 en-us 规则
                for chunk_bytes in self.kokoro.synthesize_stream(text, voice=voice, lang="en-us"):
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    
                    # Kokoro 现在返回的是 float32 (每个采样 4 字节), 24000Hz
                    chunk_duration = (len(chunk_bytes) / 4) / 24000
                    total_audio_len += chunk_duration
                    yield chunk_bytes
            
            elif language == "ms":
                # MMS 一次性生成
                audio = self.mms.synthesize(text, language="ms")
                first_chunk_time = time.time()
                
                # MMS 采样率是 16000
                total_audio_len = len(audio) / 16000
                yield audio.tobytes()
            
            # 打印汇总报表
            end_time = time.time()
            total_duration = end_time - start_time
            ttfb = (first_chunk_time - start_time) if first_chunk_time else 0
            final_rtf = total_duration / total_audio_len if total_audio_len > 0 else 0
            
            print(f"✅ [TTS] {language.upper()} Request complete!")
            print(f"📊 Summary ({language.upper()}): TTFB: {ttfb:.2f}s | Total: {total_duration:.2f}s | Audio: {total_audio_len:.2f}s | Final RTF: {final_rtf:.2f}")

        except Exception as e:
            print(f"❌ [TTS] {language.upper()} Error: {e}")
            raise

_service = None
def get_service():
    global _service
    if _service is None:
        _service = UnifiedTTSService()
    return _service
