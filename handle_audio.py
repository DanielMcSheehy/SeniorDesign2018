import librosa
import os.path
import numpy as np
import torch
import random
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

    #Returns np.ndarray
    def compute_mfccs(self, data):
        data = librosa.feature.mfcc(
            data,
            sr = self.sr,
            hop_length = self.hop_length,
            n_fft = self.n_fft,
            n_mfcc = self.n_mfcc
        )
        # If extracted audio shape is not exact, pad it with zeros:
        if data.shape[0] != 10 or data.shape[1] != 49: 
            length = 49 - data.shape[1]
            data = np.concatenate((data, np.zeros((10,length))), axis = 1)
        return data

    def split_data_set(self, data, training_size_percentage, 
                        testing_size_percentage, validation_size_percentage):
        training_set_size = round(training_size_percentage*len(data))
        testing_set_size = round(testing_size_percentage*len(data))
        validation_set_size = round(testing_size_percentage*len(data))

        training_set = data[:training_set_size]
        testing_set = data[training_set_size: testing_set_size]
        validation_set = data[:validation_set_size]
        return training_set, testing_set, validation_set

    
    def convert_to_minibatches(self, data, batch_size):
        batch_list = []
        label_list = []
        for item in data: 
            batch_list.append(item['input'])
            label_list.append(item['label'])
        if batch_size == 1:
            return batch_list, label_list
        else: 
            mini_batch_list = [batch_list[i * batch_size:(i + 1) * batch_size] for i in range((len(batch_list) + batch_size - 1) // batch_size )] 
            mini_batch_label = [label_list[i * batch_size:(i + 1) * batch_size] for i in range((len(label_list) + batch_size - 1) // batch_size )] 
            return mini_batch_list, mini_batch_label

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

        data = []
        label = {}
        for i, word in enumerate(wanted_words):
            label[word] = ground_truth_vector[:,i]
        print(label)
        for root, directories, files in os.walk(path):
            for directory in directories:
                if directory in wanted_words:
                    print(path+ "/"+ directory)
                    for root, directories, files in os.walk(path + "/" + directory):
                        for file in files: 
                            audio, _ = self.load_audio_file(path + "/" + directory+"/"+file)
                            input_obj = {
                                "input": self.compute_mfccs(audio),
                                "label": label[directory],
                            }
                            data.append(input_obj)
        random.shuffle(data)
        return data, label

    def get_size_of_mfcc_output(self, data):
        return Variable(IntTensor(data)).size()