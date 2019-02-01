import librosa
import numpy as np
from torch import IntTensor
from torch.autograd import Variable

class AudioPreprocessor(object):
    def __init__(self, n_fft=1012, hop_length=256, n_mfcc=20):
            super().__init__()
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.n_mfcc = n_mfcc

    def load_audio_file(self, path):
        y, sr = librosa.load(path)
        return y,sr

    def compute_mfccs(self, data, sr):
        data = librosa.feature.mfcc(
            data,
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mfcc = self.n_mfcc
        )
        print(Variable(IntTensor(data)).size())
        return data
    
    def get_size_of_mfcc_output(self, data):
        return Variable(IntTensor(data)).size()