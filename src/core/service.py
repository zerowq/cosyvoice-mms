"""
统一TTS服务核心逻辑
"""
import os
import hashlib
import shutil
from pathlib import Path
from typing import Literal, Optional

from ..config import config as global_config
from ..engines.cosyvoice_engine import CosyVoiceEngine
from ..engines.mms_engine import MMSEngine

class UnifiedTTSService:
    """自动路由: 英文 -> CosyVoice, 马来文 -> MMS"""
    
    def __init__(self, config=None):
        self.config = config or global_config
        self._cosyvoice = None
        self._mms = None
        
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
    
    def _get_cache_path(self, text: str, language: str, voice: str) -> Path:
        key = hashlib.md5(f"{text}_{language}_{voice}".encode()).hexdigest()
        return Path(self.config.cache_dir) / f"{key}.wav"
    
    def synthesize(self, text: str, language: Literal["en", "ms"], voice: Optional[str] = None, output_path: Optional[str] = None, use_cache: bool = True) -> dict:
        if voice is None:
            voice = self.config.default_english_voice if language == "en" else self.config.default_malay_voice
        
        if use_cache:
            cache_path = self._get_cache_path(text, language, voice)
            if cache_path.exists():
                if output_path:
                    shutil.copy(cache_path, output_path)
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
        return res

    def stream_synthesize(self, text: str, language: Literal["en", "ms"], voice: Optional[str] = None):
        """流式合成主入口"""
        if voice is None:
            voice = self.config.default_english_voice if language == "en" else self.config.default_malay_voice
        
        if language == "en":
            # CosyVoice 原生支持流式
            yield from self.cosyvoice.synthesize_stream(text, voice)
        elif language == "ms":
            # MMS 暂不支持流式，我们一次性生成并模拟流式返回以保持 API 兼容
            audio = self.mms.synthesize(text, language="ms")
            yield audio.tobytes()
        else:
            raise ValueError(f"Unsupported language: {language}")

_service = None
def get_service():
    global _service
    if _service is None:
        _service = UnifiedTTSService()
    return _service
