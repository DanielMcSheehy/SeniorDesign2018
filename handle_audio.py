import librosa
import numpy as np
from torch import IntTensor
from torch.autograd import Variable

class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_fft=1012, hop_length=327, n_mfcc=10): #! TODO (327->320)
            super().__init__()
            self.sr = sr
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.n_mfcc = n_mfcc

    def load_audio_file(self, path):
        y, sr = librosa.load(path, sr=self.sr)
        return y,sr

    def compute_mfccs(self, data, sr):
        data = librosa.feature.mfcc(
            data,
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mfcc = self.n_mfcc
        )
        return data
    
    def get_size_of_mfcc_output(self, data):
        return Variable(IntTensor(data)).size()