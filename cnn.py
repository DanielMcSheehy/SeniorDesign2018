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
        conv1_relu = self.conv1(traning_data).clamp(min=0) #Relu
        
        conv2_relu = self.conv2(conv1_relu).clamp(min=0) #Relu
        pool_layer = self.pool1(conv2_relu) 
        reshaped_pool = pool_layer.view(-1, 30 * 2 * 15)

        linear_layer = self.linear1(reshaped_pool)
        fc_layer = self.fully_connected(linear_layer)
        return fc_layer
