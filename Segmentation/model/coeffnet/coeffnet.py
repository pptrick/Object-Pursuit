import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
from model.coeffnet.config.deeplab_param import deeplab_param
from model.coeffnet.deeplab_block.resnet import resnet18
from model.coeffnet.deeplab_block.aspp import ASPP
from model.coeffnet.deeplab_block.decoder import Decoder
from model.coeffnet.hypernet import HypernetBlock

class Coeffnet(nn.Module):
    def __init__(self, z_dim, param_dict=deeplab_param):
        super(Coeffnet, self).__init__()
        self.param_dict = param_dict
        self.z_dim = z_dim
        self.hypernet_dict = self._construct_hypernets()
        self.z = nn.Parameter(torch.randn(z_dim))
        
    def _construct_hypernets(self):
        hypernet_dict = collections.OrderedDict()
        for param in self.param_dict:
            shape = self.param_dict[param]
            hypernet_dict[param.replace('.', '-')] = HypernetBlock(self.z_dim, kernel_size=shape[2], in_size=shape[1], out_size=shape[0])
        return nn.ModuleDict(hypernet_dict)
    
    def _gen_weights(self, z, hypernet_dict):
        weights = collections.OrderedDict()
        for param in hypernet_dict:
            weight_param = param.replace('-', '.')
            weights[weight_param+'.weight'], weights[weight_param+'.bn_weight'], weights[weight_param+'.bn_bias'] = hypernet_dict[param](z)
        return weights
    
    def forward(self, input):
        z = self.z
        weights = self._gen_weights(z, self.hypernet_dict)
        # backbone forward
        x, low_level_feat = resnet18("backbone", input, weights, output_stride=16)
        # aspp forward
        x = ASPP("aspp", x, weights, output_stride=16)
        # decoder forward
        x = Decoder("decoder", x, low_level_feat, weights)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x