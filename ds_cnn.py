import torch
# Conv -> 4XDWS_CONV -> AVG pool
# 1x25x64
class DS_CNNnet(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        """
        Initalize a Convolutional Neural Network with multiple layers. 
        Includes Relu's and pooling. 
        """
        

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
     
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

    def forward(self, traning_data):
        conv1_relu = self.conv1(traning_data).clamp(min=0) #Relu

        conv2_relu = self.conv2(conv1_relu).clamp(min=0) #Relu
        pool_layer = self.pool1(conv2_relu) 
        reshaped_pool = pool_layer.view(-1, 30 * 2 * 15)

        linear_layer = self.linear1(reshaped_pool)
        fc_layer = self.fully_connected(linear_layer)
        return fc_layer
