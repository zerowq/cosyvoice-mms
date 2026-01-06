"""
CosyVoice TTS 引擎封装（支持 v2.0 和 v3.0）
"""
import os
import sys
import torch
import torchaudio
import tempfile
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Generator

# 获取根目录
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
COSYVOICE_PATH = ROOT_DIR / "CosyVoice"

# 动态添加路径以便导入 CosyVoice
if str(COSYVOICE_PATH) not in sys.path:
    sys.path.insert(0, str(COSYVOICE_PATH))
    sys.path.insert(0, str(COSYVOICE_PATH / "third_party" / "Matcha-TTS"))


def preprocess_prompt_audio(audio_path: str, target_sr: int = 16000, max_val: float = 0.8) -> str:
    """
    预处理参考音频（与 Docker 镜像中的 postprocess 逻辑一致）
    1. 去除静音
    2. 音量归一化
    3. 添加尾部静音
    返回处理后的临时文件路径
    """
    # 加载音频
    speech, sr = torchaudio.load(audio_path)
    speech = speech.mean(dim=0, keepdim=True)  # 转为单声道

    # 重采样到目标采样率
    if sr != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(speech)

    # 转为 numpy 进行 librosa 处理
    speech_np = speech.numpy().flatten()

    # 1. 去除静音 (trim silence)
    speech_trimmed, _ = librosa.effects.trim(speech_np, top_db=60, frame_length=440, hop_length=220)

    # 2. 音量归一化
    speech_tensor = torch.from_numpy(speech_trimmed).unsqueeze(0)
    if speech_tensor.abs().max() > max_val:
        speech_tensor = speech_tensor / speech_tensor.abs().max() * max_val

    # 3. 添加尾部静音 (0.2秒)
    tail_silence = torch.zeros(1, int(target_sr * 0.2))
    speech_tensor = torch.cat([speech_tensor, tail_silence], dim=1)

    # 保存到临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    torchaudio.save(temp_file.name, speech_tensor, target_sr)

    return temp_file.name

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
        processed_audio = None
        try:
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
                    # 预处理参考音频（trim silence, normalize）
                    processed_audio = preprocess_prompt_audio(ref_audio_path)
                    iterable = model.inference_cross_lingual(text, processed_audio, stream=stream)
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
        finally:
            # 清理临时文件
            if processed_audio and os.path.exists(processed_audio):
                os.unlink(processed_audio)

        audio = torch.cat(audio_chunks, dim=1) if len(audio_chunks) > 1 else audio_chunks[0]

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torchaudio.save(output_path, audio, self.sample_rate)
        return audio

    def synthesize_stream(self, text: str, voice: str = "英文女") -> Generator[bytes, None, None]:
        """
        流式合成音频块逻辑内容 (日志已由上层统一处理)
        """
        processed_audio = None
        try:
            model = self.model
            spk_list = model.list_available_spks()

            if voice in spk_list:
                iterable = model.inference_sft(text, voice, stream=True)
            else:
                voice_dir = os.path.join(ROOT_DIR, "static", "voices")
                ref_audio_path = os.path.join(voice_dir, f"{voice.strip()}.wav")

                if os.path.exists(ref_audio_path):
                    print(f"🎤 [CosyVoice] Using reference audio: {os.path.basename(ref_audio_path)}")
                    # 预处理参考音频（trim silence, normalize）
                    processed_audio = preprocess_prompt_audio(ref_audio_path)
                    iterable = model.inference_cross_lingual(text, processed_audio, stream=True)
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
        finally:
            # 清理临时文件
            if processed_audio and os.path.exists(processed_audio):
                os.unlink(processed_audio)
