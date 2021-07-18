import torch
import torch.nn as nn
import torch.nn.functional as F
from model.coeffnet.deeplab_block.function import *

def last_conv(name, x, params):
    x = conv2d(x, name+".0", params, stride=1, padding=1, bias=None)
    x = batch_norm(x, name+".1", params)
    x = relu(x)
    x = dropout(x, 0.5)
    x = conv2d(x, name+".4", params, stride=1, padding=1, bias=None)
    x = batch_norm(x, name+".5", params)
    x = relu(x)
    x = dropout(x, 0.1)
    x = conv2d(x, name+".8", params, stride=1)
    return x

def Decoder(name, x, low_level_feat, params):
    low_level_feat = conv2d(low_level_feat, name+".conv1", params, bias=None)
    low_level_feat = batch_norm(low_level_feat, name+".bn1", params)
    low_level_feat = relu(low_level_feat)
    
    x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
    x = torch.cat((x, low_level_feat), dim=1)
    x = last_conv(name+".last_conv", x, params)
    
    return x