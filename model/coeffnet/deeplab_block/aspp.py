import torch
import torch.nn as nn
import torch.nn.functional as F
from model.coeffnet.deeplab_block.function import *


def _ASPPModule(name, x, params, padding, dilation):
    # atrous_conv
    x = conv_layer(x, name+".atrous_conv", params, padding=padding, dilation=dilation, stride=1, bias=None)
    # relu
    x = relu(x)
    return x
    
def global_avg_pool(name, x, params):
    # AdaptiveAvgPool2d
    x = F.adaptive_avg_pool2d(x, (1,1))
    # Conv2d
    x = conv_layer(x, name + ".1", params, bias=None, stride=1)
    # relu
    x = relu(x)
    return x
    
    
def ASPP(name, x, params, output_stride=16):
    if output_stride == 16:
        dilations = [1, 6, 12, 18]
    elif output_stride == 8:
        dilations = [1, 12, 24, 36]
    else:
        raise NotImplementedError
    x1 = _ASPPModule(name+".aspp1", x, params, padding=0, dilation=dilations[0])
    x2 = _ASPPModule(name+".aspp2", x, params, padding=dilations[1], dilation=dilations[1])
    x3 = _ASPPModule(name+".aspp3", x, params, padding=dilations[2], dilation=dilations[2])
    x4 = _ASPPModule(name+".aspp4", x, params, padding=dilations[3], dilation=dilations[3])
    x5 = global_avg_pool(name+".global_avg_pool", x, params)
    x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
    x = torch.cat((x1, x2, x3, x4, x5), dim=1)
    
    # conv1
    x = conv_layer(x, name + ".conv1", params, bias=None)
    # relu
    x = relu(x)
    # dropout
    x = dropout(x, p=0.5)
    return x
    
        