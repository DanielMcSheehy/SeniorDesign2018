import torch
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
y, sr = audio_manager.load_audio_file('./example_audio/example_audio.wav')
test = audio_manager.compute_mfccs(y, sr)
#print(test)

example_audio = torch.from_numpy(test)
print(audio_manager.get_size_of_mfcc_output(test)) #torch.Size([20, 2584])
train(model, example_audio, 1, 50, example_truth_vector, 1e-4)

