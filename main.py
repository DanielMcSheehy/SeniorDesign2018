from cnn import CNNnet
from train import train, test
import torch

#! Batch size: 28
example_batch = torch.randn(28,1,10,49)
# Have to be a long one demensional tensor (For CEL)
y_example_truth_vector = torch.autograd.Variable(torch.LongTensor(28).random_(5))
# Construct our model by instantiating the class defined above
model = CNNnet()

#     net,    batch,  batch_size,n_epochs, truth_vector,   learning_rate
train(model, example_batch, 28, 500, y_example_truth_vector, 1e-4)

test(model, example_batch, y_example_truth_vector)