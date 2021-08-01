import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
from model.coeffnet.config.deeplab_param import deeplab_param
from model.coeffnet.deeplab_block.resnet import resnet18
from model.coeffnet.deeplab_block.aspp import ASPP
from model.coeffnet.deeplab_block.decoder import Decoder
from model.coeffnet.hypernet import Hypernet

class Coeffnet(nn.Module):
    n_channels = 3
    n_classes = 1
    def __init__(self, z_dim, param_dict=deeplab_param):
        super(Coeffnet, self).__init__()
        self.z_dim = z_dim
        self.z = nn.Parameter(torch.randn(z_dim))
        self.hypernet = Hypernet(z_dim, param_dict=param_dict)
    
    def forward(self, input):
        z = self.z
        weights = self.hypernet(z)
        # backbone forward
        x, low_level_feat = resnet18("backbone", input, weights, output_stride=16)
        # aspp forward
        x = ASPP("aspp", x, weights, output_stride=16)
        # decoder forward
        x = Decoder("decoder", x, low_level_feat, weights)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x