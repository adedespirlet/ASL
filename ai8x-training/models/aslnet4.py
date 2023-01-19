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


##
"""
Network description class
"""
class AslNet(nn.Module):

    def __init__(self, num_classes=24, num_channels=1, dimensions=(28, 28),
                    planes=128, pool=2, fc_inputs=12, bias=False, **kwargs):
        super().__init__()
    # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, planes, 3,
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions -> MNIST: 28x28 | CIFAR: 32x32

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(planes, 60, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> MNIST: 14x14 | CIFAR: 16x16
        if pad == 2:
            dim += 2  # MNIST: padding 2 -> 16x16 | CIFAR: padding 1 -> 16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(60, 56, 3,
                                                 pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 8x8
        # padding 1 -> no change in dimensions

        self.conv4 = ai8x.FusedAvgPoolConv2dReLU(56, fc_inputs, 3,
                                                 pool_size=pool, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= pool  # pooling, padding 0 -> 4x4
        # padding 1 -> no change in dimensions

        self.fc = ai8x.SoftwareLinear(fc_inputs*dim*dim, num_classes, bias=True)
        #self.fc = ai8x.FusedLinearReLU(fc_inputs*dim*dim,num_classes,bias=True) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

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

