import torch
import torch.nn as nn
import collections
from torch.nn.parameter import Parameter

def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.net(input)   
    
class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, input):
        return self.net(input)
    
class HypernetFCBlock(nn.Module):
    def __init__(self, z_dim, kernel_size, in_size, out_size):
        super(HypernetFCBlock, self).__init__()
        self.z_dim = z_dim
        self.kernel_size = kernel_size
        self.in_size = in_size
        self.out_size = out_size
        
        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.out_size*self.kernel_size*self.kernel_size)),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.in_size*self.out_size*self.kernel_size*self.kernel_size)),2))
        
        self.w_bn1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size)),2))
        self.b_bn1 = Parameter(torch.fmod(torch.randn((self.out_size)),2))

        self.w_bn2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size)),2))
        self.b_bn2 = Parameter(torch.fmod(torch.randn((self.out_size)),2))
        
    def forward(self, z):
        h_final = torch.matmul(z, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.kernel_size, self.kernel_size)
        
        bn_weight = torch.matmul(z, self.w_bn1) + self.b_bn1
        bn_bias = torch.matmul(z, self.w_bn2) + self.b_bn2
        
        return kernel, bn_weight, bn_bias

class HypernetConvBlock(nn.Module):
    def __init__(self, z_dim, kernel_size, in_size, out_size):
        super(HypernetConvBlock, self).__init__()
        self.init_block = 32
        self.kernel_size = kernel_size
        self.in_size = in_size
        self.out_size = out_size
        self.z_dim = z_dim
        self.expand_linear = nn.Linear(in_features=z_dim, out_features=self.init_block*self.init_block)
        self.conv_kernel_gen = self._make_layers(self.init_block, self.init_block, out_size, in_size, kernel_size)
        self.bn_weight, self.bn_bias = self._make_bn(z_dim, out_size)
    
    def _make_bn(self, z_dim, out_size):
        weight = FCBlock(hidden_ch=out_size, num_hidden_layers=1, in_features=z_dim, out_features=out_size, outermost_linear=True)
        bias = FCBlock(hidden_ch=out_size, num_hidden_layers=1, in_features=z_dim, out_features=out_size, outermost_linear=True)
        return weight, bias
        
    def _make_layers(self, init_h, init_w, H, W, Ker):
        h, w = init_h, init_w
        old_layer = 1
        hidden_layer = 4
        proc = []
        while H>h or W>w:
            if H>h and W>w:
                proc.append(self._upsample((2,2)))
                proc.append(self._ident_conv(old_layer, hidden_layer, kernel=3))
                proc.append(nn.LeakyReLU())
                old_layer = hidden_layer
                hidden_layer *= 2
                h *= 2
                w *= 2
            elif H>h:
                proc.append(self._upsample((2,1)))
                proc.append(self._ident_conv(old_layer, hidden_layer, kernel=5))
                proc.append(nn.LeakyReLU())
                old_layer = hidden_layer
                hidden_layer *= 2
                h *= 2
            elif W>w:
                proc.append(self._upsample((1,2)))
                proc.append(self._ident_conv(old_layer, hidden_layer, kernel=5))
                proc.append(nn.LeakyReLU())
                old_layer = hidden_layer
                hidden_layer *= 2
                w *= 2
        
        proc.append(self._channels_conv(old_layer, Ker*Ker, kernel=5))
        proc.append(nn.LeakyReLU())
        proc.append(self._resize_conv(Ker*Ker, h-H, w-W))
        return nn.Sequential(*proc)
            
    def _upsample(self, factor=(2,2)):
        return nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
    
    def _ident_conv(self, in_channels, out_channels, kernel=3):
        assert (kernel+1)%2 == 0
        padding_size = (int((kernel-1)/2), int((kernel-1)/2))
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=1, padding=padding_size, padding_mode='zeros')
    
    def _channels_conv(self, in_channels, out_channels, kernel=3):
        assert (kernel+1)%2 == 0
        padding_size = (int((kernel-1)/2), int((kernel-1)/2))
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=1, padding=padding_size, padding_mode='zeros')
    
    def _resize_conv(self, channels, delta_h, delta_w):
        kernel_size = (delta_h+1, delta_w+1)
        return nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1)
    
    def forward(self, z):
        out = self.expand_linear(z)
        out = out.view(1,1,self.init_block,self.init_block)
        out = self.conv_kernel_gen(out)
        out = out.permute(2,3,0,1).view(self.out_size, self.in_size, self.kernel_size, self.kernel_size)
        bn_w = self.bn_weight(z)
        bn_b = self.bn_bias(z)
        return out, bn_w, bn_b
        
if __name__ == "__main__":
    z_dim = 100
    z = torch.randn(z_dim)
    block = HypernetConvBlock(z_dim, kernel_size=1, in_size=1280, out_size=256)
    block(z)
    # print(out.size(), w.size(), b.size())