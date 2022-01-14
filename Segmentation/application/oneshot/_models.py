# select model for n-shot learning
from model.deeplabv3.deeplab import *
from model.unet import UNet
from model.coeffnet.coeffnet import Coeffnet, Singlenet

def select_model(model,
                 device,
                 z_dim=100,
                 base_dir=None,
                 pretrained_hypernet=None,
                 pretrained_backbone=None,
                 use_backbone=True):
    if model == "unet":
        net = UNet(n_channels=3, n_classes=1, bilinear=True)
    elif model == "deeplab":
        net = DeepLab(num_classes = 1, backbone = 'resnetsub', output_stride = 16, freeze_backbone=False, pretrained_backbone=False)
        if use_backbone:
            net.init_backbone(pretrained_backbone, freeze=True)
    elif model == "singlenet": # directly pursuit a 100-dim z instead of the combination of bases
        net = Singlenet(z_dim=z_dim, device=device, use_backbone=use_backbone)
        net.init_hypernet(pretrained_hypernet, freeze=True)
        if use_backbone:
            net.init_backbone(pretrained_backbone, freeze=True)
    elif model == "coeffnet": # combine bases and pursuit coeffs
        assert base_dir is not None
        net = Coeffnet(base_dir=base_dir, 
                       z_dim=z_dim, 
                       device=device, 
                       hypernet_path=pretrained_hypernet, 
                       backbone_path=pretrained_backbone,
                       use_backbone=use_backbone)
        # hypernet and backbone are initialized inside the coeffnet
    else:
        raise NotImplementedError
    
    net.to(device=device)
    return net