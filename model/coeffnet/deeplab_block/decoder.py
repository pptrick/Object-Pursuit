import torch
import torch.nn as nn
import torch.nn.functional as F
from model.coeffnet.deeplab_block.function import *

def last_conv(name, x, params):
    x = conv_layer(x, name+".0", params, stride=1, padding=1, bias=None)
    x = relu(x)
    x = dropout(x, 0.5)
    x = conv_layer(x, name+".4", params, stride=1, padding=1, bias=None)
    x = relu(x)
    x = dropout(x, 0.1)
    # special layer
    x = conv2d(x, name+".8", params, stride=1, bias=True)
    return x

def Decoder(name, x, low_level_feat, params):
    low_level_feat = conv_layer(low_level_feat, name+".conv1", params, bias=None)
    low_level_feat = relu(low_level_feat)
    
    x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
    x = torch.cat((x, low_level_feat), dim=1)
    x = last_conv(name+".last_conv", x, params)
    
    return x