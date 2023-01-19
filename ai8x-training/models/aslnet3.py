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

        self.conv1 = ai8x.FusedConv2dReLU(in_channels = num_channels, out_channels = 32, kernel_size = 3,
                                          padding=0, bias=bias, **kwargs) #sixe 26x26
        # padding 1 -> no change in dimensions
        dim_x -= 2  # pooling, padding 0
        dim_y -= 2
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 32, out_channels = 128, kernel_size = 3,
                                          padding=0, bias=bias, **kwargs) #13x13 after pooling and 11 afterconv  #pool size default is 2x2

        dim_x //= 2  # pooling, padding 0
        dim_y //= 2

        dim_x -= 2  # pooling, padding 0
        dim_y -= 2
    
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 128, out_channels = 512, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) #after max 6 and after conv 4 
                
        self.mp= ai8x.MaxPool2d(kernel_size=2, **kwargs)  #2x2
       
        self.fc1 = ai8x.FusedLinearReLU(2048, 1024, bias=True, **kwargs) #512x2x2

        self.fc2 = ai8x.FusedLinearReLU(in_features=1024, out_features=256, bias=True, **kwargs)
        
        self.dropout = nn.Dropout(0.5)
        self.fc3 = ai8x.FusedLinearReLU(in_features=256, out_features=25, wide=True, bias=True, **kwargs)

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
        
        x = self.conv3(x)
       
        x = self.mp(x)

        x = x.view(x.size(0), -1)
    
        x = self.fc1(x)
        x = self.fc2(x)
        x=self.dropout(x)
        x = self.fc3(x)
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

