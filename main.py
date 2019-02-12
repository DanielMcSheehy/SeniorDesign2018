import torch
import np
from cnn import CNNnet
from ds_cnn import DS_CNNnet
from train import train, test
from handle_audio import AudioPreprocessor

# Construct our model by instantiating the class defined above
model = CNNnet()

audio_manager = AudioPreprocessor()
wanted_words = ['on', 'off', 'stop']

path_to_dataset = '/Users/dsm/Downloads/speech_commands_v0.01'
data, labelDictionary = audio_manager.extract_audio_files(path_to_dataset, wanted_words)

training_set, testing_set, validation_set = audio_manager.split_data_set(data, .80, .10, .10)

mini_batch_list, mini_batch_label = audio_manager.convert_to_minibatches(training_set, 64)

# needed to reshape/organize training set: 
#! Not converting it to minibatch, just reorganizing the data
training_list, training_label_list = audio_manager.convert_to_minibatches(testing_set, 1)

# needed to reshape/organize validation set: 
validation_list, validation_label_list = audio_manager.convert_to_minibatches(validation_set, 1)

num_epochs = 100
for epoch in range(num_epochs):
    for i, batch in enumerate(mini_batch_list):
        train(model, batch, 64,mini_batch_label[i], 1e-4)

    if epoch % 10 == 0: 
        test(model, testing_set, training_label_list)

print("Final validation of model:")
test(model, validation_set, label)

