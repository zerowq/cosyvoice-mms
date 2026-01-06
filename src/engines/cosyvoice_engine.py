"""
CosyVoice TTS 引擎封装（支持 v2.0 和 v3.0）
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
    """CosyVoice TTS引擎（自动检测 v2.0 或 v3.0）"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._loaded = False
        # 根据模型路径判断版本
        self._is_v3 = "CosyVoice3" in model_path or "Fun-CosyVoice" in model_path

    def _load_model(self):
        if not self._loaded:
            try:
                import torch
                # 只有在 CUDA 可用时才开启 fp16
                use_fp16 = torch.cuda.is_available()

                if self._is_v3:
                    from cosyvoice.cli.cosyvoice import CosyVoice3
                    print(f"🔄 Loading CosyVoice 3.0 on {self.device} (fp16={use_fp16})...")
                    self._model = CosyVoice3(
                        self.model_path,
                        load_trt=False,
                        fp16=use_fp16
                    )
                    print(f"✅ CosyVoice 3.0 loaded on {self.device} (fp16={use_fp16})!")
                else:
                    from cosyvoice.cli.cosyvoice import CosyVoice2
                    print(f"🔄 Loading CosyVoice 2.0 on {self.device} (fp16={use_fp16})...")
                    self._model = CosyVoice2(
                        self.model_path,
                        load_jit=True,
                        load_trt=False,
                        fp16=use_fp16
                    )
                    print(f"✅ CosyVoice 2.0 loaded on {self.device} (fp16={use_fp16})!")

                self._loaded = True
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

            if voice in spk_list:
                iterable = model.inference_sft(text, voice, stream=stream)
            else:
                # 容错：去除前后空格
                clean_voice = voice.strip().replace(" ", "").lower()
                ref_audio_path = os.path.join(ROOT_DIR, "static", "voices", f"{voice.strip()}.wav")

                if not os.path.exists(ref_audio_path):
                    alt_path = os.path.join(ROOT_DIR, "static", "voices", f"{clean_voice}.wav")
                    if os.path.exists(alt_path):
                        ref_audio_path = alt_path

                if os.path.exists(ref_audio_path):
                    print(f"🎤 Using local reference audio: {ref_audio_path}")
                    # 直接传递路径字符串，CosyVoice 内部会处理加载
                    iterable = model.inference_cross_lingual(text, ref_audio_path, stream=stream)
                else:
                    if spk_list:
                        print(f"⚠️ Voice '{voice}' not found, fallback to '{spk_list[0]}'")
                        iterable = model.inference_sft(text, spk_list[0], stream=stream)
                    else:
                        raise ValueError(f"Voice '{voice}' not found and no reference audio at static/voices/{voice}.wav")

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

            # 推理逻辑选择（完全恢复 master 分支逻辑）
            if voice in spk_list:
                iterable = model.inference_sft(text, voice, stream=True)
            else:
                voice_dir = os.path.join(ROOT_DIR, "static", "voices")
                ref_audio_path = os.path.join(voice_dir, f"{voice.strip()}.wav")

                if os.path.exists(ref_audio_path):
                    print(f"🎤 [CosyVoice] Using reference audio: {os.path.basename(ref_audio_path)}")
                    iterable = model.inference_cross_lingual(text, ref_audio_path, stream=True)
                else:
                    print(f"⚠️ [CosyVoice] Voice '{voice}' not found, falling back to English default")
                    iterable = model.inference_sft(text, "英文女", stream=True)

            # 流式迭代
            for chunk in iterable:
                speech = chunk['tts_speech'].numpy().flatten()
                yield speech.tobytes()

        except Exception as e:
            print(f"❌ [CosyVoice] Streaming error: {e}")
            raise
