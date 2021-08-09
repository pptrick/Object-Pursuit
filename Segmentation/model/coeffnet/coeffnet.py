import collections
import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.coeffnet.config.deeplab_param import deeplab_param
from model.coeffnet.deeplab_block.resnet import resnet18
from model.coeffnet.deeplab_block.aspp import ASPP
from model.coeffnet.deeplab_block.decoder import Decoder
from model.coeffnet.hypernet import Hypernet

def deeplab_forward(input, weights):
    # backbone forward
    x, low_level_feat = resnet18("backbone", input, weights, output_stride=16)
    # aspp forward
    x = ASPP("aspp", x, weights, output_stride=16)
    # decoder forward
    x = Decoder("decoder", x, low_level_feat, weights)
    x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
    return x

class Singlenet(nn.Module):
    n_channels = 3
    n_classes = 1
    def __init__(self, z_dim, device, param_dict=deeplab_param):
        super(Singlenet, self).__init__()
        self.z_dim = z_dim
        self.device = device
        self.z = nn.Parameter(torch.randn(z_dim))
        self.hypernet = Hypernet(z_dim, param_dict=param_dict)
        
    def save_z(self, file_path):
        with torch.no_grad():
            z = self.z.clone().detach()
            weights = self.hypernet(z)
            torch.save({'z':z, 'weights':weights}, file_path)
            
    def load_z(self, file_path):
        with torch.no_grad():
            self.z.data = torch.load(file_path, map_location=self.device)['z']
    
    def forward(self, input):
        z = self.z
        weights = self.hypernet(z)
        return deeplab_forward(input, weights)
    

class Multinet(nn.Module):
    n_channels = 3
    n_classes = 1
    def __init__(self, obj_num, z_dim, device, param_dict=deeplab_param):
        super(Multinet, self).__init__()
        self.obj_num = obj_num
        self.z_dim = z_dim
        self.device = device
        self.z = nn.Parameter(torch.randn((obj_num, z_dim)))
        self.hypernet = Hypernet(z_dim, param_dict=param_dict)
        
    def forward(self, input, ident):
        ident = ident[0].item()
        z = self.z[ident]
        weights = self.hypernet(z)
        return deeplab_forward(input, weights)

    
class Coeffnet(nn.Module):
    n_channels = 3
    n_classes = 1
    def __init__(self, base_dir, z_dim, device, hypernet_path=None, param_dict=deeplab_param, nn_init=True):
        super(Coeffnet, self).__init__()
        self.z_dim = z_dim
        self.device = device
        self.base_dir = base_dir
        
        # base & coeffs
        self.zs, self.base_num = self._get_z_bases(base_dir, device)
        # self.coeffs = nn.Parameter(torch.tensor([-0.00848394 , 0.35106376 ,-0.10851663 , 0.9120488 ]))
        if nn_init:
            self.coeffs = nn.Parameter(torch.randn(self.base_num))
            self.init_value = 1.0/math.sqrt(self.base_num)
            torch.nn.init.constant_(self.coeffs, self.init_value)
        else:
            self.coeffs = nn.Parameter(torch.randn(self.base_num))
            
        # forward
        self.combine_func = self._linear
        self.hypernet = Hypernet(z_dim, param_dict=param_dict)
        self._init_hypernet(hypernet_path)
        
    def _get_z_bases(self, base_dir, device):
        base_files = [os.path.join(base_dir, file) for file in sorted(os.listdir(base_dir)) if file.endswith(".json")]
        print("Base files: ", base_files)
        zs = []
        for f in base_files:
            z = torch.load(f, map_location=device)['z']
            assert(z.size()[0] == self.z_dim)
            zs.append(z)
        base_num = len(zs)
        return zs, base_num
    
    def _linear(self, zs, coeffs):
        assert(len(zs)>0 and len(zs)==coeffs.size()[0])
        z = zs[0] * coeffs[0]
        for i in range(1, len(zs)):
            z += zs[i] * coeffs[i]
        return z
    
    def _init_hypernet(self, hypernet_path):
        if hypernet_path is not None:
            print(hypernet_path)
            assert(os.path.isfile(hypernet_path) and hypernet_path.endswith(".pth"))
            state_dict = torch.load(hypernet_path, map_location=self.device)
            hypernet_dict = collections.OrderedDict()
            for param in state_dict:
                if param.startswith("hypernet."):
                    new_param = re.match(r'hypernet\.(.+)', param).group(1)
                    hypernet_dict[new_param] = state_dict[param]
                elif param.startswith("blocks."):
                    hypernet_dict[param] = state_dict[param]
            self.hypernet.load_state_dict(hypernet_dict)
        # freeze hypernet
        for param in self.hypernet.parameters():
            param.requires_grad = False
    
    def forward(self, input):
        new_z = self.combine_func(self.zs, self.coeffs)
        weights = self.hypernet(new_z)
        return deeplab_forward(input, weights)