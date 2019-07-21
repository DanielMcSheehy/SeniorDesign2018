import torch
import np
import sound_augmentation as sound_augmentation
import sounddevice as sd
import librosa
import time
from ds_cnn import DS_CNNnet
from handle_audio import AudioPreprocessor
from IPython.display import Audio

audio_manager = AudioPreprocessor()
background_audio = sound_augmentation.load_background_audio()

# white_bg_noise = audio_manager.load_audio_file('/Users/dsm/code/SeniorDesign/SeniorDesign2018/_background_noise_/white_noise.wav')[0]
on_audio, sr = librosa.load('/Users/dsm/Downloads/Data_Samples_with_extra/Luz/female10mexico16.wav', duration=1)
sd.play(on_audio, sr)
time.sleep(3)

result = sound_augmentation.add_background_noise(on_audio, background_audio)
sd.play(result, sr)
time.sleep(3)

result = sound_augmentation.shift(on_audio, True)
sd.play(result, sr)
time.sleep(3)

# result = sound_augmentation.add_reverb(result)
# sd.play(result, sr)
# time.sleep(3)

