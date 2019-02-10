import torch
import np
from cnn import CNNnet
from ds_cnn import DS_CNNnet
from train import train, test
from handle_audio import AudioPreprocessor


#! Batch size: 64
example_batch = torch.randn(64,1,10,49)


# Have to be a long one demensional tensor (For CEL)
example_truth_vector = torch.autograd.Variable(torch.LongTensor(64).random_(5))

# Construct our model by instantiating the class defined above
model = CNNnet()

audio_manager = AudioPreprocessor()
wanted_words = ['on', 'off', 'stop']
# Data: array of {"input": tensor(10,41), "label": One hot encoded}
# Label: label[word] = one hot encoded vector

path_to_dataset = '/Users/dsm/Downloads/speech_commands_v0.01'
data, labelDictionary = audio_manager.extract_audio_files(path_to_dataset, wanted_words)

training_set, testing_set, validation_set = audio_manager.split_data_set(data, .80, .10, .10)

mini_batch_list, mini_batch_label = audio_manager.convert_to_minibatches(training_set, 64)

# needed to reshape/organize training set: 
training_list, training_label_list = audio_manager.convert_to_minibatches(testing_set, len(testing_set))

num_epochs = 100
for epoch in range(num_epochs):
    for i, batch in enumerate(mini_batch_list):
        batch = torch.from_numpy(np.array(batch)).float()
        batch = batch[:, None, :, :]
        label = torch.stack(mini_batch_label[i]).long()
        train(model, batch, 64,label, 1e-4)

    if epoch % 10 == 0: 
        testing_set = torch.from_numpy(np.array(training_list[0])).float()
        testing_set = testing_set[:, None, :, :]
        label = torch.stack(training_label_list[0]).long()
        test(model, testing_set, label)

print("Final validation of model:")
test(model, validation_set, label)
m
