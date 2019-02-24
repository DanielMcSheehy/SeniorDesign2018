import torch
import np
from ds_cnn import DS_CNNnet
from handle_audio import AudioPreprocessor
from IPython.display import Audio
import sounddevice as sd
import librosa

audio_manager = AudioPreprocessor()


# Construct our model by instantiating the class defined above
#model = CNNnet()
# model = DS_CNNnet()
on_label = [1,0,0]
label = ['on', 'off', 'stop']

white_bg_noise = audio_manager.load_audio_file('/Users/dsm/code/SeniorDesign/SeniorDesign2018/_background_noise_/white_noise.wav')
y, sr = librosa.load('/users/dsm/code/SeniorDesign/SeniorDesign2018/example_audio/example_on.wav')
on_audio = audio_manager.load_audio_file('/users/dsm/code/SeniorDesign/SeniorDesign2018/example_audio/example_on.wav')[0]
white_bg_noise = audio_manager.load_audio_file('/Users/dsm/code/SeniorDesign/SeniorDesign2018/_background_noise_/white_noise.wav')[0]
result = 0.99 * y + 0.01 * white_bg_noise[:22050]
# Ask about sr
sd.play(result, sr)
d
#Audio(result)
