import re
from torch import tensor
import torch
import torch.nn.functional as F

def conv2d(x, name, params, bias=None, stride=1, padding=0, dilation=1):
    return F.conv2d(x, params[name+".weight"], bias=bias, stride=stride, padding=padding, dilation=dilation)

def batch_norm(x, name, params):
    return _batch_norm_origin(x, name, params)
    
def _batch_norm_origin(x, name, params):
    running_mean, running_var =  params[name+".running_mean"].clone().detach(), params[name+".running_var"].clone().detach()
    res =  F.batch_norm(x, running_mean, running_var,
                    weight=params[name+".weight"], bias=params[name+".bias"], training=True)
    return res
    
def _batch_norm_linear(x, name, params):
    device = params[name+".running_mean"].device
    size = params[name+".running_mean"].size()
    init_mean = torch.zeros(size).to(device)
    init_var = torch.ones(size).to(device)
    return F.batch_norm(x, init_mean, init_var,
                    weight=params[name+".weight_prime"], bias=params[name+".bias_prime"])
    
def relu(x):
    return F.relu(x)

def dropout(x, p=0.5):
    return F.dropout(x, p=p)