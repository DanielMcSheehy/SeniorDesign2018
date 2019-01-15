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

        self.linear1 = torch.nn.Linear(30 * 2 * 15, 28) 

        # torch.nn.Conv2d(1, 10, kernel_size=3, stride=1),
        # torch.nn.ReLU(),
        # torch.nn.Conv2d(10, 20, kernel_size=3, stride=1),
        # torch.nn.Linear(5, D_out),

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.conv1(x).clamp(min=0)
        #print(h_relu.size())
        y_pred = self.conv2(h_relu).clamp(min=0)  #28,28,1,1 -> 28,28,4,10!!

        pooled = self.pool1(y_pred) #[28, 30, 2, 15]
       
        pooled = pooled.view(-1, 30 * 2 * 15)
        lin = self.linear1(pooled)
        print(lin.size())
        return lin




# Create random Tensors to hold inputs and outputs
#x = torch.randn(10, 49)
x = torch.randn(28,1,10,49)
y = torch.randn(28,28)

# Construct our model by instantiating the class defined above
model = CNNnet()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
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