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

#     net,    batch,  batch_size,n_epochs, truth_vector,   learning_rate
# train(model, example_batch, 64, 50, example_truth_vector, 1e-4)

# test(model, example_batch, example_truth_vector)

# # test Depthwise seperable convolutional network

# train(DS_CNNnet(), example_batch, 64, 50, example_truth_vector, 1e-4)

# test(DS_CNNnet(), example_batch, example_truth_vector)





audio_manager = AudioPreprocessor()
wanted_words = ['on', 'off', 'stop']
# Data: array of {"input": tensor(10,41), "label": One hot encoded}
# Label: label[word] = one hot encoded vector
data, labelDictionary = audio_manager.extract_audio_files('/Users/dsm/Downloads/speech_commands_v0.01', wanted_words)

mini_batch_list, mini_batch_label = audio_manager.convert_to_minibatches(data, 64)

for i, batch in enumerate(mini_batch_list):
    print("Training")
    batch = torch.from_numpy(np.array(batch)).float()
    batch = batch[:, None, :, :]
    label = torch.stack(mini_batch_label[i]).long()
    train(model, batch, 64, 50, label, 1e-4)
d



# y, sr = audio_manager.load_audio_file('./example_audio/example_audio.wav')
# t = audio_manager.compute_mfccs(y, sr)
# #print(test)

# audio_manager.extract_audio_files()

# example_audio = torch.from_numpy(t)
# print(audio_manager.get_size_of_mfcc_output(t))

# tensor = example_audio[None, None, :, :] #Size: (1, 10, 49)
# tensor = tensor.expand(64, 1, 10, 49)


# train(model, tensor.float(), 64, 50, example_truth_vector, 1e-4)



# batch_array = []
# for _ in range(64):
#     batch_array.append(example_audio[None, None, :, :].float())
# test(model, batch_array, example_truth_vector)

# test(model, tensor, )