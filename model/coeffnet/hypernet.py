import torch
import torch.nn as nn
import collections

from model.coeffnet.config.deeplab_param import *
from model.coeffnet.hypernet_block import HypernetConvBlock
    
class Hypernet(nn.Module):
    def __init__(self, z_dim, param_dict=deeplab_param):
        super(Hypernet, self).__init__()
        self.param_dict = param_dict
        self.z_dim = z_dim
        self.blocks = self._construct_blocks()
        
    def _construct_blocks(self):
        hypernet_dict = collections.OrderedDict()
        for param in self.param_dict:
            shape = self.param_dict[param]
            hypernet_dict[param.replace('.', '-')] = HypernetConvBlock(self.z_dim, kernel_size=shape[2], in_size=shape[1], out_size=shape[0])
        return nn.ModuleDict(hypernet_dict)
    
    def forward(self, z):
        weights = collections.OrderedDict()
        for param in self.blocks:
            weight_param = param.replace('-', '.')
            weights[weight_param+'.weight'], weights[weight_param+'.bn_weight'], weights[weight_param+'.bn_bias'] = self.blocks[param](z)
        return weights
    
