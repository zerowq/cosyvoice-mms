"""
CosyVoice 2.0 TTS 引擎封装
"""
import os
import sys
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Generator

# 获取根目录
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
COSYVOICE_PATH = ROOT_DIR / "CosyVoice"

# 动态添加路径以便导入 CosyVoice
if str(COSYVOICE_PATH) not in sys.path:
    sys.path.insert(0, str(COSYVOICE_PATH))
    sys.path.insert(0, str(COSYVOICE_PATH / "third_party" / "Matcha-TTS"))

class CosyVoiceEngine:
    """CosyVoice 2.0 英文TTS引擎"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._loaded = False
        
    def _load_model(self):
        if not self._loaded:
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2
                import torch
                # 只有在 CUDA 可用时才开启 fp16，Mac(MPS)和CPU环境下保持 False 以确保绝对稳定
                use_fp16 = torch.cuda.is_available()
                
                print(f"🔄 Loading CosyVoice 2.0 on {self.device} (fp16={use_fp16})...")
                self._model = CosyVoice2(
                    self.model_path,
                    load_jit=True,
                    load_trt=False,
                    fp16=use_fp16
                )
                self._loaded = True
                print(f"✅ CosyVoice 2.0 loaded on {self.device} (fp16={use_fp16})!")
            except Exception as e:
                print(f"❌ Failed to load CosyVoice: {e}")
                raise
        return self._model
    
    @property
    def model(self):
        return self._load_model()
    
    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate
    
    def list_voices(self) -> list:
        return self.model.list_available_spks()
    
    def synthesize(self, text: str, voice: str = "英文女", output_path: Optional[str] = None, stream: bool = False) -> torch.Tensor:
        audio_chunks = []
        try:
            # 复用 synthesize_stream 的逻辑来获取生成器 (注意：流式返回的是 bytes，这里需要改一下或者重新实现逻辑)
            # 为了简单，我们手动复制一下逻辑，但这次不转 bytes
            
            model = self.model
            spk_list = model.list_available_spks()

            if spk_list and voice in spk_list:
                iterable = model.inference_sft(text, voice, stream=stream)
            elif spk_list:
                print(f"⚠️ Voice '{voice}' not found, using preset voice: {spk_list[0]}")
                iterable = model.inference_sft(text, spk_list[0], stream=stream)
            else:
                # 没有预设音色，直接使用 inference_sft（CosyVoice 会使用默认音色）
                print(f"⚠️ No preset voices available, using default voice via inference_sft")
                iterable = model.inference_sft(text, voice, stream=stream)

            for result in iterable:
                audio_chunks.append(result['tts_speech'])
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"CosyVoice inference failed: {e}")
        
        audio = torch.cat(audio_chunks, dim=1) if len(audio_chunks) > 1 else audio_chunks[0]
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torchaudio.save(output_path, audio, self.sample_rate)
        return audio

    def synthesize_stream(self, text: str, voice: str = "英文女") -> Generator[bytes, None, None]:
        """
        流式合成音频块逻辑内容 (日志已由上层统一处理)
        """
        try:
            model = self.model
            spk_list = model.list_available_spks()

            # 尝试使用预设音色
            if spk_list and voice in spk_list:
                print(f"🎤 [CosyVoice] Using preset voice: {voice}")
                iterable = model.inference_sft(text, voice, stream=True)
            elif spk_list:
                # 使用第一个可用的预设音色
                default_voice = spk_list[0]
                print(f"⚠️ [CosyVoice] Voice '{voice}' not found, using preset voice: {default_voice}")
                iterable = model.inference_sft(text, default_voice, stream=True)
            else:
                # 没有预设音色，直接使用 inference_sft（CosyVoice 会使用默认音色）
                print(f"⚠️ [CosyVoice] No preset voices available, using default voice via inference_sft")
                iterable = model.inference_sft(text, voice, stream=True)

            # 流式迭代
            for chunk in iterable:
                speech = chunk['tts_speech'].numpy().flatten()
                yield speech.tobytes()

        except Exception as e:
            print(f"❌ [CosyVoice] Streaming error: {e}")
            raise
