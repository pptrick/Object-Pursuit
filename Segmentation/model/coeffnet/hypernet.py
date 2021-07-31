import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class HypernetBlock(nn.Module):
    def __init__(self, z_dim, kernel_size, in_size, out_size):
        super(HypernetBlock, self).__init__()
        self.z_dim = z_dim
        self.kernel_size = kernel_size
        self.in_size = in_size
        self.out_size = out_size
        
        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.kernel_size*self.kernel_size)),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.kernel_size*self.kernel_size)),2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)),2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)),2))
        
        self.w_bn1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size)),2))
        self.b_bn1 = Parameter(torch.fmod(torch.randn((self.out_size)),2))

        self.w_bn2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size)),2))
        self.b_bn2 = Parameter(torch.fmod(torch.randn((self.out_size)),2))
        
        
    def forward(self, z):
        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.kernel_size, self.kernel_size)
        
        bn_weight = torch.matmul(z, self.w_bn1) + self.b_bn1
        bn_bias = torch.matmul(z, self.w_bn2) + self.b_bn2
        
        return kernel, bn_weight, bn_bias