##################################################################################################
# ASL network
# Aurore De Spirlet
# 2022 - ETH Zurich
###################################################################################################
"""
ASL network description
"""
from signal import pause
from torch import nn

import ai8x

import matplotlib
import matplotlib.pyplot as plt

"""
Network description class
"""
class AslNet(nn.Module):
    """
    7-Layer CNN - Lightweight image classification
    """
    def __init__(self, num_classes=10, dimensions=(28, 28), num_channels=1, bias=False, **kwargs):
        super().__init__()

        # assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim_x, dim_y = dimensions

        self.conv1 = ai8x.FusedConv2dReLU(in_channels = num_channels, out_channels = 64, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) #sixe 26x26

        self.conv2 = ai8x.FusedConv2dReLU(64, out_channels = 64, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) #sixe 26x26
        # padding 1 -> no change in dimensions
        dim_x -= 2  # pooling, padding 0
        dim_y -= 2
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 64, out_channels = 128, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) #14x14 after pooling and 11 afterconv  #pool size default is 2x2
        #
        self.conv4 = ai8x.FusedConv2dReLU(in_channels = 128, out_channels = 128, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) 
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 128, out_channels = 256, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) #7x7 after pooling and 11 afterconv  #pool size default is 2x2

        self.conv6 = ai8x.FusedConv2dReLU(in_channels = 256, out_channels = 256, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) #sixe 
        self.conv7 = ai8x.FusedConv2dReLU(in_channels = 256, out_channels = 256, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) #sixe 
        self.conv8 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 256, out_channels = 512, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) #3x3 after pooling and 11 afterconv  #pool size default is 2x2
        self.conv9 = ai8x.FusedConv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) #3x3
        
        self.ap= ai8x.AvgPool2d(kernel_size=2, stride=None, **kwargs)

        self.fc1 = ai8x.Linear(512*1, 100, bias=True, **kwargs) #512x2x2


        dim_x //= 2  # pooling, padding 0
        dim_y //= 2

        dim_x -= 2  # pooling, padding 0
        dim_y -= 2
    
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    """
    Assemble the model
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # # Data plotting - for debug
        # matplotlib.use('MacOSX')
        # plt.imshow(x[1, 0], cmap="gray")
        # plt.show()
        # breakpoint()
        
        x = self.conv1(x)
    
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.conv4(x)
        #print(x.shape)
        x= self.conv5(x)
        #print(x.shape)

        x= self.conv6(x)
        #print(x.shape)

        x=  self.conv7(x)
        #print(x.shape)
        x = self.conv8(x)
        #print(x.shape)
        x = self.conv9(x)
        #print(x.shape)
        x= self.ap(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
       
        # Loss chosed, CrossEntropyLoss, takes softmax into account already func.log_softmax(x, dim=1))

        return x

def aslnet(pretrained=False, **kwargs):
    """
    Constructs an asl model.
    """
    assert not pretrained
    return AslNet(**kwargs)

"""
Network description
"""
models = [
    {
        'name': 'aslnet',
        'min_input': 1,
        'dim': 2,
    }
]

