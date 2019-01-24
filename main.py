from cnn import CNNnet
from train import train, test
import torch

#! Batch size: 64
example_batch = torch.randn(64,1,10,49)
# Have to be a long one demensional tensor (For CEL)
example_truth_vector = torch.autograd.Variable(torch.LongTensor(64).random_(5))
# Construct our model by instantiating the class defined above
model = CNNnet()

#     net,    batch,  batch_size,n_epochs, truth_vector,   learning_rate
train(model, example_batch, 64, 50, example_truth_vector, 1e-4)

test(model, example_batch, example_truth_vector)