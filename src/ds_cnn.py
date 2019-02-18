import torch
# Conv -> 4XDWS_CONV -> AVG pool
# 1x25x64
class DS_CNNnet(torch.nn.Module):
    def __init__(self):
        """
        Initalize a Convolutional Neural Network with multiple layers. 
        Includes Relu's and pooling. 
        """
        super(DS_CNNnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 28, kernel_size=(4,10), stride=1, padding=0)

        self.conv2 = torch.nn.Conv2d(28,28, kernel_size=1, stride = 1,padding = 0, dilation = 1, groups=28)
        self.pointwise2 = torch.nn.Conv2d(28,28,1,1,0,1,1)

        self.conv3 = torch.nn.Conv2d(28,28, kernel_size=1,stride = 1,padding = 0,  dilation = 1, groups=28)
        self.pointwise3 = torch.nn.Conv2d(28,28,1,1,0,1,1,)

        self.conv4 = torch.nn.Conv2d(28,28, kernel_size=1,stride = 1,padding = 0,  dilation = 1, groups=28)
        self.pointwise4 = torch.nn.Conv2d(28,28,1,1,0,1,1)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.linear1 = torch.nn.Linear(28*3*20, 16) 
        self.fully_connected = torch.nn.Linear(16, 3) 
    
    def forward(self, traning_data):
        conv1_relu = self.conv1(traning_data).clamp(min=0) #Relu

        conv2_relu = self.conv2(conv1_relu).clamp(min=0) #Relu
        pointwise2 = self.conv2(conv2_relu).clamp(min=0)
        
        conv3_relu = self.conv2(pointwise2).clamp(min=0) #Relu
        pointwise3 = self.conv2(conv3_relu).clamp(min=0)

        conv4_relu = self.conv2(pointwise3).clamp(min=0) #Relu
        pointwise4 = self.conv2(conv4_relu).clamp(min=0)
        
        pool_layer = self.pool1(pointwise4) 
        reshaped_pool = pool_layer.view(-1, 28*3*20)

        linear_layer = self.linear1(reshaped_pool)
        fc_layer = self.fully_connected(linear_layer)
        return fc_layer


