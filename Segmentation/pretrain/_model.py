from model.coeffnet.coeffnet import *

class Multinet(nn.Module):
    n_channels = 3
    n_classes = 1
    def __init__(self, classes, z_dim, use_backbone=True, freeze_backbone=True):
        super(Multinet, self).__init__()
        self.classes = classes
        self.z_dim = z_dim
        self.z = nn.Parameter(torch.randn((classes, z_dim))) # each object has a representation z
        if use_backbone:
            self.hypernet = Hypernet(z_dim, param_dict=deeplab_param_decoder)
        else:
            self.hypernet = Hypernet(z_dim, param_dict=deeplab_param)
        self.use_backbone = use_backbone
        
        if use_backbone:
            self.backbone = build_backbone("resnetsub", 16, nn.BatchNorm2d, pretrained=True)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        
    def forward(self, input, ident):
        z = self.z[ident]
        # hypernet predict weights
        weights = self.hypernet(z)
        if not self.use_backbone:
            # forward
            return deeplab_forward(input, weights), z
        else:
            # backbone forward
            x, low_level_feat = self.backbone(input)
            # decoder forward
            return deeplab_forward_no_backbone(input, x, low_level_feat, weights), z
        
def get_multinet(class_num, z_dim, device, use_backbone=True, freeze_backbone=True):
    net = Multinet(class_num, z_dim, use_backbone, freeze_backbone)
    net.to(device=device)
    return net