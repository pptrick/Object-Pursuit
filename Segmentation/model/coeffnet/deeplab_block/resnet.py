import torch
import torch.nn as nn
import torch.nn.functional as F
from model.coeffnet.deeplab_block.function import *

EXPANSION = 1

def BasicBlock(name, x, params, stride=1, dilation=1, downsample=None):
    residual = x
    
    out = conv_layer(x, name+".conv1", params, stride=stride, dilation=dilation, padding=dilation, bias=False)
    out = relu(out, inplace=True)
    
    out = conv_layer(out, name+".conv2", params, dilation=dilation, padding=dilation, bias=False)
    
    if downsample is not None:
        residual = downsample(name+".downsample", x, params, stride=stride)
        
    out += residual
    out = relu(out, inplace=True)
    
    return out

def downsample(name, x, params, stride, bias=False):
    out = conv_layer(x, name+".0", params, stride=stride, bias=bias)
    return out

def layer(name, x, params, inplanes, block, planes, blocks, stride=1, dilation=1):
    down_sample = None
    if stride != 1 or inplanes != planes * EXPANSION:
        down_sample = downsample
    
    out = block(name+".0", x, params, stride=stride, dilation=dilation, downsample=down_sample)
    inplanes = inplanes * EXPANSION
    for i in range(1, blocks):
        out = block(name+"."+str(i), out, params, dilation=dilation)
        
    return out

def MG_unit(name, x, params, inplanes, block, planes, blocks, stride=1, dilation=1):
    down_sample = None
    if stride != 1 or inplanes != planes * EXPANSION:
        down_sample = downsample
        
    out = block(name+".0", x, params, stride=stride, dilation=blocks[0]*dilation, downsample=down_sample)
    inplanes = inplanes * EXPANSION
    for i in range(1, len(blocks)):
        out = block(name+"."+str(i), out, params, stride=1, dilation=blocks[i]*dilation)
    
    return out
    

def ResNet(name, input, params, block, layers, output_stride):
    inplanes = 64
    blocks = [1,2,4]
    if output_stride == 16:
        strides = [1, 2, 2, 1]
        dilations = [1, 1, 1, 2]
    elif output_stride == 8:
        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
    else:
        raise NotImplementedError
    
    # head
    x = conv_layer(input, name+".conv1", params, bias=False, stride=2, padding=3)
    x = relu(x, inplace=True)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # body
    x = layer(name+".layer1", x, params, inplanes, block, 64, layers[0], stride=strides[0], dilation=dilations[0])
    low_level_feat = x
    x = layer(name+".layer2", x, params, inplanes, block, 128, layers[1], stride=strides[1], dilation=dilations[1])
    x = layer(name+".layer3", x, params, inplanes, block, 256, layers[2], stride=strides[2], dilation=dilations[2])
    x = MG_unit(name+".layer4", x, params, inplanes, block, 512, blocks, stride=strides[3], dilation=dilations[3])
    
    return x, low_level_feat

def resnet18(name, input, params, output_stride):
    return ResNet(name, input, params, BasicBlock, [2,2,2,2], output_stride=output_stride)
    