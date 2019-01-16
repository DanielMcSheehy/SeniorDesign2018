from cnn import CNNnet
from train import train
import torch

x = torch.randn(28,1,10,49)
# Have to be a long one demensional tensor (For CEL)
y = torch.autograd.Variable(torch.LongTensor(28).random_(5))
# Construct our model by instantiating the class defined above
model = CNNnet()

# net, batch, batch_size, n_epochs, truth_vector, learning_rate
train(model, x, 1, 500, y, 1e-4)