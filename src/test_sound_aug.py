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


# Construct our model by instantiating the class defined above
#model = CNNnet()
# model = DS_CNNnet()
on_label = [1,0,0]
label = ['on', 'off', 'stop']

white_bg_noise = audio_manager.load_audio_file('/Users/dsm/code/SeniorDesign/SeniorDesign2018/_background_noise_/white_noise.wav')[0]
on_audio, sr = librosa.load('/users/dsm/code/SeniorDesign/SeniorDesign2018/example_audio/example_on.wav')
result = sound_augmentation.add_background_noise(on_audio)
sd.play(result, sr)
time.sleep(3)

result = sound_augmentation.shift(result)
sd.play(result, sr)
time.sleep(3)

result = sound_augmentation.add_reverb(result)
sd.play(result, sr)
time.sleep(3)

