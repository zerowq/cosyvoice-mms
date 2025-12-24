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
        流式合成音频块
        支持预设音色或通过本地 wav 文件进行 zero-shot 克隆
        """
        model = self.model
        spk_list = model.list_available_spks()
        
        # 1. 预处理：去除前后空格及中间空格
        clean_voice = voice.strip().replace(" ", "").lower()
        
        # 2. 如果是预设音色，直接使用 SFT 推理
        if voice in spk_list:
            iterable = model.inference_sft(text, voice, stream=True)
            
        # 3. 如果不是预设，但在 static/voices 下有同名 wav 文件，则进行 Zero-Shot 推理
        else:
            voice_dir = os.path.join(ROOT_DIR, "static", "voices")
            ref_audio_path = os.path.join(voice_dir, f"{voice.strip()}.wav")
            
            # 容错匹配：如果直接找找不到，遍历目录进行松散匹配 (忽略空格和大小写)
            if not os.path.exists(ref_audio_path) and os.path.exists(voice_dir):
                for f in os.listdir(voice_dir):
                    if f.lower().endswith(".wav"):
                        f_name = f.rsplit('.', 1)[0]
                        if f_name.replace(" ", "").lower() == clean_voice:
                            ref_audio_path = os.path.join(voice_dir, f)
                            break

            if os.path.exists(ref_audio_path):
                print(f"🎤 Using local reference audio: {ref_audio_path}")
                # 直接传递路径字符串
                iterable = model.inference_cross_lingual(text, ref_audio_path, stream=True)
            else:
                 # 最后的兜底：如果连文件都没有，尝试用第一个预设（如果有）或报错
                if spk_list:
                    print(f"⚠️ Voice '{voice}' not found, fallback to '{spk_list[0]}'")
                    iterable = model.inference_sft(text, spk_list[0], stream=True)
                else:
                    raise ValueError(f"Voice '{voice}' not found and no reference audio at static/voices/{voice}.wav")

        for result in iterable:
            audio_tensor = result['tts_speech']
            yield audio_tensor.cpu().numpy().tobytes()
