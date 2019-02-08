import librosa
import os.path
import numpy as np
import torch
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

    def compute_mfccs(self, data):
        data = librosa.feature.mfcc(
            data,
            sr=self.sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mfcc = self.n_mfcc
        )
        return data
    

    def to_one_hot(self, y, depth=None):
        y_tensor = y.data if isinstance(y, Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        depth = depth if depth is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], depth).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
        return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

    def extract_audio_files(self, path, wanted_words):
        ground_truth_vector = torch.arange(0,len(wanted_words))
        ground_truth_vector= self.to_one_hot(ground_truth_vector)
        print(ground_truth_vector)

        data = []
        label = {}
        for i, word in enumerate(wanted_words):
            label[word] = ground_truth_vector[:,i]
        print(label)
        for root, directories, files in os.walk(path):
            for directory in directories:
                if directory in wanted_words:
                    print(path+ "/"+ directory)
                    for root, directories, files in os.walk(path+ "/"+ directory):
                        for file in files: 
                            input_obj = {
                                "input": self.compute_mfccs(file),
                                "label": label[directory],
                            data.append(input_obj)
        return data

    def get_size_of_mfcc_output(self, data):
        return Variable(IntTensor(data)).size()