from cnn import CNNnet
from train import train
import torch

example_batch = torch.randn(28,1,10,49)
# Have to be a long one demensional tensor (For CEL)
y_example_truth_vector = torch.autograd.Variable(torch.LongTensor(28).random_(5))
# Construct our model by instantiating the class defined above
model = CNNnet()

#     net,    batch,  batch_size,n_epochs, truth_vector,   learning_rate
train(model, example_batch, 1, 500, y_example_truth_vector, 1e-4)