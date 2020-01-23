import torch

class DNNnet(torch.nn.Module):
    def __init__(self, output_size): 
        """
        Initalize a Convolutional Neural Network with multiple layers. 
        Includes Relu's and pooling. 
        """
        super(DNNnet, self).__init__()
        self.fc1 = torch.nn.Linear(25*10, 144)
        self.fc2 = torch.nn.Linear(144, 144)
        self.fc3 = torch.nn.Linear(144, 144)
        self.fully_connected = torch.nn.Linear(144, output_size) 

    def forward(self, traning_data):
        reshaped_traning_data = traning_data.view(-1, 25*10)
        fc1_relu = self.fc1(reshaped_traning_data).clamp(min=0) #Relu
        fc2_relu = self.fc2(fc1_relu).clamp(min=0) #Relu
        fc3_relu = self.fc3(fc2_relu).clamp(min=0) #Relu
        fc_layer = self.fully_connected(fc3_relu)
        return fc_layer

