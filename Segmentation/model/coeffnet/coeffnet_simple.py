import os
import re
import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.coeffnet.coeffnet import deeplab_forward_no_backbone, deeplab_forward
from model.deeplabv3.backbone import build_backbone

def init_backbone(model_path, backbone, device, freeze=False):
    '''init backbone with pretrained model'''
    if model_path is not None and os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location=device)
        backbone_dict = collections.OrderedDict()
        for param in state_dict:
            if param.startswith("backbone."):
                new_param = re.match(r'backbone\.(.+)', param).group(1)
                backbone_dict['module.'+new_param] = state_dict[param]
            elif param.startswith("module."):
                backbone_dict[param] = state_dict[param]
        if len(backbone_dict) > 0:
            backbone.load_state_dict(backbone_dict)
        else:
            print("[Warning] init backbone failed")
    # freeze backbone
    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
            
def init_hypernet(model_path, hypernet, device, freeze=False):
    '''init hypernet with pretrained model'''
    if model_path is not None and os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location=device)
        hypernet_dict = collections.OrderedDict()
        for param in state_dict:
            if param.startswith("hypernet."):
                new_param = re.match(r'hypernet\.(.+)', param).group(1)
                hypernet_dict[new_param] = state_dict[param]
            elif param.startswith("blocks."):
                hypernet_dict[param] = state_dict[param]
        if len(hypernet_dict) > 0:
            hypernet.load_state_dict(hypernet_dict)
        else:
            print("[Warning] init hypernet failed")
    # freeze hypernet
    if freeze:
        for param in hypernet.parameters():
            param.requires_grad = False


class Backbone(nn.Module):
    def __init__(self, Type="resnetsub", output_stride=16, pretrained=True):
        super(Backbone, self).__init__()
        self.module = build_backbone(backbone=Type, output_stride=output_stride, BatchNorm=nn.BatchNorm2d, pretrained=pretrained)
        
    def forward(self, input):
        return self.module(input)
        

class Singlenet(nn.Module):
    n_channels = 3
    n_classes = 1
    def __init__(self, z_dim):
        super(Singlenet, self).__init__()
        self.z_dim = z_dim
        self.z = nn.Parameter(torch.randn(z_dim)) # parameter
        
    def load_z(self, file_path):
        with torch.no_grad():
            if os.path.isfile(file_path):
                z = torch.load(file_path, map_location=self.z.device)['z']
                if z.size() == self.z.size():
                    self.z.data = z
                else:
                    print(f"[Warning] parameter z in singlenet has the size {self.z.size()}, however the loaded z has the size {z.size()}")
            else:
                raise IOError
            
    def save_z(self, file_path, hypernet=None):
        with torch.no_grad():
            z = self.z.clone().detach()
            if hypernet is not None:
                weights = hypernet(z)
                torch.save({'z':z, 'weights':weights}, file_path)
            else:
                torch.save({'z':z}, file_path)
        
    def forward(self, input, hypernet, backbone=None):
        z = self.z
        weights = hypernet(z)
        if backbone is not None:
            x, low_level_feat = backbone(input)
            out = deeplab_forward_no_backbone(input, x, low_level_feat, weights)
        else:
            out = deeplab_forward(input, weights)
        return out
    
    def L1_loss(self, coeff):
        return coeff * F.l1_loss(self.z, torch.zeros(self.z.size()).to(self.z.device))
    

class Coeffnet(nn.Module):
    n_channels = 3
    n_classes = 1
    def __init__(self, bases_num, nn_init=True):
        super(Coeffnet, self).__init__()
        self.base_num = bases_num
        self.coeffs = nn.Parameter(torch.randn(self.base_num))
        if nn_init:
            init_value = 1.0/math.sqrt(self.base_num)
            torch.nn.init.constant_(self.coeffs, init_value)
            
        self.combine_func = self._linear
            
    def _linear(self, zs, coeffs):
        assert(len(zs)>0 and len(zs)==coeffs.size()[0])
        z = zs[0] * coeffs[0]
        for i in range(1, len(zs)):
            z += zs[i] * coeffs[i]
        return z
    
    def forward(self, input, bases_z, hypernet, backbone=None):
        new_z = self.combine_func(bases_z, self.coeffs)
        weights = hypernet(new_z)
        if backbone is not None:
            x, low_level_feat = backbone(input)
            out = deeplab_forward_no_backbone(input, x, low_level_feat, weights)
        else:
            out = deeplab_forward(input, weights)
        return out
    
    def L1_loss(self, coeff):
        return coeff * F.l1_loss(self.coeffs, torch.zeros(self.coeffs.size()).to(self.coeffs.device))
