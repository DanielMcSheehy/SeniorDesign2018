import librosa
import os
import os.path
import numpy as np
import torch
import random
from torch import IntTensor
from torch.autograd import Variable
import sound_augmentation as sound_augmentation

class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_fft=1012, hop_length=325, n_mfcc=10): 
        """Audio preprocessor interface that can upload downloaded audio files,
            do feature extraction (mfcc), and reorganize data to be used in 
            neural networks

        Note:
            The default attributes are to make sure audio features extraction
            stays organized and consistant, do not alter. 

            Output size of data should be (10, 49)
            (16000/325 = 49, x 10 features (n_mfcc))

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
        self.window = 'hamming'
        self.win_length = 400 # 0.025*16000
        self.fmin = 20
        self.fmax = 4000
        self.n_mels = 40

    def load_audio_file(self, path):
        """Loads audio file from downloaded audio file path 

        Args:
            path (str): Location of folder with .wav files.

        Returns:
            y (bytes): encoded audio file.

        """
        y, sr = librosa.load(path, sr=self.sr)
        return y,sr

    def feature_extraction(self, data):
        extracted_data = []
        for audio in data: 
            extracted_data.append(
                {
                    'input': self.compute_mfccs(audio['input']),
                    'label': audio['label'],
                }
            )
        return extracted_data

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
        # Test computational load of this or Hamming feature extraction:s
        D = np.abs(librosa.stft(data, window=self.window, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length))**2
        S = librosa.feature.melspectrogram(S=D, y=data, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        extracted_features = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=self.n_mfcc)
        # extracted_features = librosa.feature.mfcc(
        #     data,
        #     sr = self.sr,
        #     hop_length = self.hop_length,
        #     n_fft = self.n_fft,
        #     n_mfcc = self.n_mfcc
        # )

        # If extracted audio shape is not exact, pad it with zeros:
        if extracted_features.shape[1] < 49: 
            length = 49 - extracted_features.shape[1]
            extracted_features = np.concatenate((extracted_features, np.zeros((10,length))), axis = 1)
        elif extracted_features.shape[1] > 49: 
             extracted_features = extracted_features[:][:,:-1]
        return extracted_features

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

        # If data is not an array, converts it to one. 
        if not type(data) in (tuple, list):
            data = [data]

        for item in data: 
            batch = torch.from_numpy(np.array(item['input'])).float()
            batch = batch[None, :, :]
            label = item['label']
            batch_list.append(batch)
            label_list.append(label)
        stacked_batchs = torch.stack(batch_list)
        stacked_labels = torch.stack(label_list)
        # Instead of array of sized batches, just returns entire stacked tensor
        if batch_size == 1: 
            return stacked_batchs, stacked_labels
        else: 
            batch_list = torch.split(stacked_batchs, batch_size)
            label_list = torch.split(stacked_labels, batch_size)
            return batch_list, label_list

    def augment_data(self, data, background_audio):
        for btx, batch in enumerate(data):
            batch['input'] = sound_augmentation.augment_sound(batch['input'], background_audio)
        # Randomize Data between Epochs:
        random.shuffle(data)
        return data

    def to_one_hot(self, y, depth=None):
        y_tensor = y.data if isinstance(y, Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        depth = depth if depth is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], depth).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
        return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot
    
    def get_single_batch(self, path, label_arr):
        audio =  self.load_audio_file(path)[0]
        input_obj = {
            "input": self.compute_mfccs(audio),
            "label": torch.tensor(label_arr),
        }
        batch, label = self.convert_to_minibatches(input_obj, 1)
        return batch, label

    def add_silence(self, data, silence_one_hot_encoded_vector, percentage_silence):
        silence_audio, _ =  self.load_audio_file('../_background_noise_/silence.wav')
        silence_amount_to_add = round((len(data)*percentage_silence)/(1-percentage_silence))
        for _ in range(silence_amount_to_add):
            silence_obj = {
                "input": silence_audio,
                "label": silence_one_hot_encoded_vector,
            }
            data.append(silence_obj)
        return data

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
                                "input": audio,
                                "label": label[directory],
                            }
                            data.append(input_obj)
        data = self.add_silence(data, label['silence'], .1)
        random.shuffle(data)
        return data, label

        def add_unknown_words(self, available_words, wanted_words):
            unknown_words = []
            for word in available_words:
                if word not in wanted_words:
                    unknown_words.append(word)
            return unknown_words #!not yet
        
        
 
    def benchmark(self):
        CPU_Pct=str(os.popen('''ps -A -o %cpu | awk '{s+=$1} END {print s "%"}' ''').readline())
        print("CPU Usage = " + CPU_Pct)