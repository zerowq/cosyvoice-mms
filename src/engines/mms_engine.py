"""
Meta MMS-TTS 引擎封装
"""
import os
import torch
import numpy as np
import scipy.io.wavfile as wav
from typing import Optional, Dict
from transformers import VitsModel, AutoTokenizer
from pathlib import Path

class MMSEngine:
    """Meta MMS-TTS 马来文引擎"""
    
    LANGUAGE_MODELS = {
        "en": "mms-tts-eng",
        "ms": "mms-tts-zlm",
        "id": "mms-tts-ind",
    }
    
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        self._models: Dict[str, VitsModel] = {}
        self._tokenizers: Dict[str, AutoTokenizer] = {}
    
    def _load_model(self, language: str):
        if language not in self._models:
            model_name = self.LANGUAGE_MODELS.get(language)
            if not model_name:
                raise ValueError(f"Unsupported language: {language}")
            
            # 优先从本地 model_dir 加载
            local_model_path = self.model_dir / model_name
            
            if local_model_path.exists():
                print(f"🔄 Loading local MMS-TTS from {local_model_path}...")
                self._models[language] = VitsModel.from_pretrained(
                    local_model_path, 
                    local_files_only=True
                ).to(self.device)
                self._tokenizers[language] = AutoTokenizer.from_pretrained(
                    local_model_path,
                    local_files_only=True
                )
            else:
                print(f"⚠️ Local model not found at {local_model_path}, trying internet...")
                self._models[language] = VitsModel.from_pretrained(f"facebook/{model_name}").to(self.device)
                self._tokenizers[language] = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
            
            print(f"✅ MMS-TTS ({language}) loaded!")
            
    def get_sample_rate(self, language: str = "ms") -> int:
        self._load_model(language)
        return self._models[language].config.sampling_rate
    
    def synthesize(self, text: str, language: str = "ms", output_path: Optional[str] = None) -> np.ndarray:
        self._load_model(language)
        model = self._models[language]
        tokenizer = self._tokenizers[language]
        
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = model(**inputs).waveform
        
        waveform = output.squeeze().cpu().numpy()
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            wav.write(output_path, rate=model.config.sampling_rate, data=waveform)
        return waveform
