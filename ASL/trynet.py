###################################################################################################
# MemeNet network
# Aurore De Spirlet
# 2022 - ETH Zurich
###################################################################################################
"""
MemeNet network description
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
    def __init__(self, num_classes=24, dimensions=(28, 28), num_channels=1, bias=False, **kwargs):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dReLU(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) # [m, 16, 28, 28]
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 16, out_channels = 24, kernel_size = 3,
                                          padding=0, bias=bias, **kwargs) # [m, 24, 14, 14]
        

        self.conv3 = ai8x.FusedConv2dReLU(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1) # [m, 32, 14, 14]
       
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 32, out_channels = 24, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs) # [m, 24, 7, 7]
        self.fc5 = ai8x.FusedLinearReLU(in_features=24*7*7, out_features=120,bias=True) # [m, 120]
        self.fc6 = ai8x.FusedLinearReLU(in_features=120, out_features=84,bias= True) # [m, 84]
        self.fc7 = ai8x.Linear(in_features=84, out_features=num_classes,wide=True,bias=True) # [m, 10]


        # assert dimensions[0] == dimensions[1]  # Only square supported
        # Keep track of image dimensions so one constructor works for all image sizes
        #dim_x, dim_y = dimensions
        #dim_x //= 2  # pooling, padding 0
        #dim_y //= 2
        # conv padding 1 -> no change in dimensions
        #self.fcx = ai8x.Linear(16, num_classes, wide=True, bias=True, **kwargs)


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
        x = self.conv4(x)

        #########################
        # TODO: Add more layers #
        #########################

        x = x.view(x.size(0), -1)

        #########################
        # TODO: Add more layers #
        #########################
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        # Loss chosed, CrossEntropyLoss, takes softmax into account already func.log_softmax(x, dim=1))
        return x


def aslnet(pretrained=False, **kwargs):
    """
    Constructs a MemeNet model.
    """
    assert not pretrained
    return AslNet(**kwargs)

"""
Network description
"""
models = [
    {
        'name': 'memenet',
        'min_input': 1,
        'dim': 2,
    }
]

