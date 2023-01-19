#################################################################################################
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


##LENET NETWORK
"""
Network description class
"""
class AslNet(nn.Module):

    def __init__(self, num_classes=24, num_channels=1, dimensions=(28, 28),
                    planes=128, pool=2, fc_inputs=12, bias=False, **kwargs):
        super().__init__()
    # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 6, 3,
                                          padding=2, bias=bias, **kwargs)  #
            
        
        self.conv2 = ai8x.FusedConv2dReLU(6, 6, 3,padding=0, bias=bias, **kwargs)  #28x28x6
      
        self.conv3 = ai8x.FusedAvgPoolConv2dReLU(6, 16, 3, pool_size=2, pool_stride=2, padding=0,
                                                 bias=bias, **kwargs)  #12x12x16

        self.conv4 = ai8x.FusedConv2dReLU(16, 16, 3, padding=0, bias=bias, **kwargs)  #10x10x16
        #
        self.mp= ai8x.AvgPool2d(kernel_size=2, **kwargs)  #5x5x16
     
        self.fc1 = ai8x.FusedLinearReLU(5*5*16, 120, bias=True, **kwargs) #512x2x2
        self.fc2 = ai8x.FusedLinearReLU(120, 84, bias=True, **kwargs) #512x2x2

        self.fc3 = ai8x.Linear(84, num_classes, wide=True, bias=True, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        
        x = self.conv2(x)
 
        x = self.conv3(x)
        
        x = self.conv4(x)
        
        x= self.mp(x)
      
        x = x.view(x.size(0), -1)
        x= self.fc1(x)
        x= self.fc2(x)
        x= self.fc3(x)
        
    
        return x

#model = Net().to(device)

#summary(model, (1, 28, 28))


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

