import torch

class DS_CNNnet(torch.nn.Module):
    def __init__(self, output_size):
        """
        Initalize a Convolutional Neural Network with multiple layers. 
        Includes Relu's and pooling. 
        """
        super(DS_CNNnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(4,10), stride=1, padding=0)

        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size = (3,3), stride = 1, padding = 1, dilation = 1, groups = 64)
        self.pointwise2 = torch.nn.Conv2d(64,64,1,1,0,1,1)

        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size = (3,3), stride = 1, padding = 1,  dilation = 1, groups = 64)
        self.pointwise3 = torch.nn.Conv2d(64,64,1,1,0,1,1,)

        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size = (3,3), stride = 1, padding = 1,  dilation = 1, groups = 64)
        self.pointwise4 = torch.nn.Conv2d(64,64,1,1,0,1,1)

        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding=0)

        self.linear1 = torch.nn.Linear(64*3*20, 16) 
        self.fully_connected = torch.nn.Linear(16, output_size) 
    
    def forward(self, traning_data):
        conv1_relu = self.conv1(traning_data).clamp(min=0) #Relu

        conv2_relu = self.conv2(conv1_relu).clamp(min=0) #Relu
        pointwise2 = self.conv2(conv2_relu)
        
        conv3_relu = self.conv2(pointwise2).clamp(min=0) #Relu
        pointwise3 = self.conv2(conv3_relu)

        conv4_relu = self.conv2(pointwise3).clamp(min=0) #Relu
        pointwise4 = self.conv2(conv4_relu)
        
        pool_layer = self.pool1(pointwise4) 
        reshaped_pool = pool_layer.view(-1, 64*3*20)

        linear_layer = self.linear1(reshaped_pool)
        fc_layer = self.fully_connected(linear_layer)
        return fc_layer


