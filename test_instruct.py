import sys
sys.path.insert(0, 'CosyVoice')
sys.path.insert(0, 'CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio

model = CosyVoice2('models/CosyVoice2-0.5B', load_jit=True, load_trt=False, fp16=True)

text = 'Hello, this is a test of English speech synthesis.'
instruct_text = 'A clear female English voice'

print("Testing inference_instruct...")
for i, chunk in enumerate(model.inference_instruct(text, '英文女', instruct_text, stream=False)):
    output_file = f'output/test_instruct_{i}.wav'
    torchaudio.save(output_file, chunk['tts_speech'], model.sample_rate)
    print(f'Saved {output_file}')
    if i >= 0:
        break

print("Done! Check output/test_instruct_0.wav")
