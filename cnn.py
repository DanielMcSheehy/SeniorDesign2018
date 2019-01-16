import torch


class CNNnet(torch.nn.Module):
    def __init__(self): #TODO: Make Network Not hardcoded
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 28, kernel_size=(4,10), stride=1, padding=0)

        self.conv2 = torch.nn.Conv2d(28, 30, kernel_size=(4,10), stride=1, padding=0)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.linear1 = torch.nn.Linear(30 * 2 * 15, 16) 
        self.fully_connected = torch.nn.Linear(16, 28) 

    def forward(self, traning_data):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        conv1_relu = self.conv1(traning_data).clamp(min=0)
        conv2_relu = self.conv2(conv1_relu).clamp(min=0)  

        pool_layer = self.pool1(conv2_relu) 
        reshaped_pool = pool_layer.view(-1, 30 * 2 * 15)

        linear_layer = self.linear1(reshaped_pool)
        fc_layer = self.fully_connected(linear_layer)
        return fc_layer


# Create random Tensors to hold inputs and outputs
#x = torch.randn(10, 49)
x = torch.randn(28,1,10,49)
# Have to be a long one demensional tensor (For CEL)
y = torch.autograd.Variable(torch.LongTensor(28).random_(5))
# Construct our model by instantiating the class defined above
model = CNNnet()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()