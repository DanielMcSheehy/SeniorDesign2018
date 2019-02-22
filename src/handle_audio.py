import librosa
import os.path
import numpy as np
import torch
import random
from torch import IntTensor
from torch.autograd import Variable

class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_fft=1012, hop_length=327, n_mfcc=10): 
        """Audio preprocessor interface that can upload downloaded audio files,
            do feature extraction (mfcc), and reorganize data to be used in 
            neural networks

        Note:
            The default attributes are to make sure audio features extraction
            stays organized and consistant, do not alter. 

            Output size of data should be (10, 49)

        Args:
            

        Attributes:
            sr (int): Sampling rate to convert audio.
            n_fft (int): 
            hop_length (int): window size that goes through audio length.
            n_mfcc(int): Number of output features in feature extractions
        """
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def load_audio_file(self, path):
        """Loads audio file from downloaded audio file path 

        Args:
            path (str): Location of folder with .wav files.

        Returns:
            y (bytes): encoded audio file.

        """
        y, sr = librosa.load(path, sr=self.sr)
        return y,sr

    def compute_mfccs(self, data):
        """Computes mfcc feature extraction of data from 
            self.load_audio_file.

        Note: 
            If window size cuts off at end, instead of having a 
            diffent size output, the output is filled in with 
            zeros to maintain 10,49 shape.

        Args:
            data (bytes): byte of loaded audio file from librosa library. 

        Returns:
            data (np.ndarray of shape (10,49)): 

        """
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
        """Splits dataset into training, testing, and validation data sets. 
            Splits it up by percents: 80%, 10%, 10% => (.8, .1, .1)

        Args:
            x_size_percentage(float): decimal of percetange of data to split into

        Returns:
            x_set (array of size (x, 10, 49)): returns total data split into percentage

        """
        training_set_size = round(training_size_percentage*len(data))
        testing_set_size = round(testing_size_percentage*len(data))
        validation_set_size = round(testing_size_percentage*len(data))

        training_set = data[:training_set_size]
        testing_set = data[training_set_size: training_set_size + testing_set_size]
        validation_set = data[:validation_set_size]
        return training_set, testing_set, validation_set

    
    def convert_to_minibatches(self, data, batch_size):
        """Converts array of Mfcc feature extracted data into
            batch_size sized batches

        Args:
            data (array of arrays with shape(10,49)): Mfcc extracted data.
            batch_size (int): Size of batches (ex: 64)

        Note: 
            If batch_size is 1, it will just return 
            all the data in one giant tensor instead of
            array of batches

        Returns:
            A string of updated factom state.

        """
        batch_list = []
        label_list = []
        for item in data: 
            batch = torch.from_numpy(np.array(item['input'])).float()
            batch = batch[None, :, :]
            label = item['label']
            batch_list.append(batch)
            label_list.append(label)
        a = batch_list
        stacked_batchs = torch.stack(batch_list)
        stacked_labels = torch.stack(label_list)
        # Instead of array of sized batches, just returns entire stacked tensor
        if batch_size == 1: 
            return stacked_batchs, stacked_labels
        else: 
            batch_list = torch.split(stacked_batchs, 64)
            label_list = torch.split(stacked_labels, 64)
            return batch_list, label_list

    def augment_data(self, batch_list, label_list, random_seed):
        augmented_batch_list = []
        for btx, batch in enumerate(batch_list):
            batch = batch + Variable(torch.randn(batch.size())* 0.2)
            augmented_batch_list.append(batch)
        #! Randomize Data between Epochs:
        #random.Random(random_seed).shuffle(augmented_batch_list)
        #random.Random(random_seed).shuffle(label_list)
        return augmented_batch_list, label_list

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