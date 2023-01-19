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


##SIMPLENETNETWORK

"""
Network description class
"""
class AslNet(nn.Module):

    def __init__(self, num_classes=24, num_channels=1, dimensions=(28, 28), bias=False, **kwargs):
        super().__init__()
    # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 16, 3, stride=1, padding=2, bias=bias,
                                            **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(16, 20, 3, stride=1, padding=2, bias=bias, **kwargs)
        self.conv3 = ai8x.FusedConv2dBNReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(20, 20, 3, pool_size=2, pool_stride=2,
                                                   stride=1, padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv7 = ai8x.FusedConv2dBNReLU(20, 44, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2dBNReLU(44, 48, 3, pool_size=2, pool_stride=2,
                                                   stride=1, padding=1, bias=bias, **kwargs)
        self.conv9 = ai8x.FusedConv2dBNReLU(48, 48, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv10 = ai8x.FusedMaxPoolConv2dBNReLU(48, 96, 3, pool_size=2, pool_stride=2,
                                                    stride=1, padding=1, bias=bias, **kwargs)
        self.conv11 = ai8x.FusedMaxPoolConv2dBNReLU(96, 512, 1, pool_size=2, pool_stride=2,
                                                    padding=0, bias=bias, **kwargs)
        self.conv12 = ai8x.FusedConv2dBNReLU(512, 128, 1, stride=1, padding=0, bias=bias, **kwargs)
        self.conv13 = ai8x.FusedMaxPoolConv2dBNReLU(128, 128, 3, pool_size=2, pool_stride=2,
                                                    stride=1, padding=1, bias=bias, **kwargs)
        self.conv14 = ai8x.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=bias,
                                  wide=True, **kwargs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = self.conv3(x)
        #print(x.size())
        x = self.conv4(x)
        #print(x.size())
        x = self.conv5(x)
        #print(x.size())
        x = self.conv6(x)
        ##print(x.size())
        x = self.conv7(x)
        #print(x.size())
        x = self.conv8(x)
        #print(x.size())
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        #print(x.size())
        x = self.conv12(x)
        #print(x.size())
        x = self.conv13(x)
        #print(x.size())
        x = self.conv14(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
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

