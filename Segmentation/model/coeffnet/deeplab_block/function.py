import re
from torch import tensor
import torch
import torch.nn.functional as F

def conv_layer(x, name, params, bias=None, stride=1, padding=0, dilation=1):
    res = conv2d(x, name, params, bias, stride, padding, dilation)
    res = batch_norm(res, name, params)
    return res

def conv2d(x, name, params, bias=None, stride=1, padding=0, dilation=1):
    bias = None if not bias else params[name+".bn_bias"]
    return F.conv2d(x, params[name+".weight"], bias=bias, stride=stride, padding=padding, dilation=dilation)
    
def batch_norm(x, name, params):
    running_mean, running_var =  params[name+".bn_bias"].clone().detach(), params[name+".bn_bias"].clone().detach() # just a place holder
    return F.batch_norm(x, running_mean, running_var,
                        weight=params[name+".bn_weight"], bias=params[name+".bn_bias"], training=True)
    
def relu(x, inplace=False):
    return F.relu(x, inplace=inplace)

def dropout(x, p=0.5):
    return F.dropout(x, p=p)