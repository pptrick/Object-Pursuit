from torch import tensor
import torch
import torch.nn.functional as F

def conv2d(x, name, params, bias=None, stride=1, padding=0, dilation=1):
    return F.conv2d(x, params[name+".weight"], bias=bias, stride=stride, padding=padding, dilation=dilation)
    
def batch_norm(x, name, params):
    running_mean, running_var =  params[name+".running_mean"].clone().detach(), params[name+".running_var"].clone().detach()
    return F.batch_norm(x, running_mean, running_var,
                    weight=params[name+".weight"], bias=params[name+".bias"], training=True)
    
def relu(x):
    return F.relu(x)

def dropout(x, p=0.5):
    return F.dropout(x, p=p)