import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from model.deeplabv3.deeplab import *

class Coeffnet_Deeplab(nn.Module):
    n_channels = 3
    n_classes = 1
    
    def __init__(self, base_dir, device):
        super(Coeffnet_Deeplab, self).__init__()
        self.base_dir = base_dir
        self.device = device
        self.base_num, self.aspp_bases, self.decoder_bases = self._parse_base_files(base_dir, device)
        print("The number of base: ", self.base_num)
        
        # self.coeffs = nn.ParameterList([nn.Parameter(torch.randn(1)) for i in range(self.base_num)])
        self.coeffs = nn.Parameter(torch.randn(self.base_num))
        self.target_net = DeepLab(num_classes=self.n_classes, backbone='resnetsub', output_stride=16, freeze_backbone=True)
        # freeze target net param
        # for param in self.target_net.parameters():
        #     param.requires_grad = False
        
    def _parse_base_files(self, base_dir, device):
        base_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir) if file.endswith(".pth")]
        base_num = len(base_files)
        aspp_weights = collections.OrderedDict()
        decoder_weights = collections.OrderedDict()
        for f in base_files:
            state_dict = torch.load(f, map_location=device)
            for param in state_dict:
                # print(param, state_dict[param].size(), type(state_dict))
                if param.startswith("backbone"):
                    pass
                elif param.startswith("aspp"):
                    param_sub = re.match(r'aspp\.(.+)', param).group(1)
                    if param_sub not in aspp_weights:
                        aspp_weights[param_sub] = [state_dict[param]]
                    else:
                        assert(state_dict[param].size() == aspp_weights[param_sub][0].size())
                        aspp_weights[param_sub].append(state_dict[param])
                elif param.startswith("decoder"):
                    param_sub = re.match(r'decoder\.(.+)', param).group(1)
                    if param_sub not in decoder_weights:
                        decoder_weights[param_sub] = [state_dict[param]]
                    else:
                        assert(state_dict[param].size() == decoder_weights[param_sub][0].size())
                        decoder_weights[param_sub].append(state_dict[param])
                else:
                    print(f"[Warning] find illegal network parameter: {param}")
        return base_num, aspp_weights, decoder_weights
    
    def _update_weights(self):
        aspp_weights = collections.OrderedDict()
        decoder_weights = collections.OrderedDict()
        for param in self.aspp_bases:
            aspp_weights[param] = self.aspp_bases[param][0] * self.coeffs[0] + self.aspp_bases[param][1] * self.coeffs[1]
        for param in self.decoder_bases:
            decoder_weights[param] = self.decoder_bases[param][0] * self.coeffs[0] + self.decoder_bases[param][1] * self.coeffs[1]
        return aspp_weights, decoder_weights
    
    def forward(self, input):
        # set network parameters:
        new_aspp, new_decoder = self._update_weights()
        self.target_net.aspp.load_state_dict(new_aspp)
        self.target_net.decoder.load_state_dict(new_decoder)
        
        # forward
        return self.target_net(input)
        
        
        