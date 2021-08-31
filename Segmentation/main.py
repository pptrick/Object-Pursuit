import torch
from object_pursuit.pursuit import pursuit

pursuit(z_dim=100, 
        data_dir="/data/pancy/iThor/single_obj/FloorPlan2",
        output_dir="./first_test",
        device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'),
        pretrained_bases=None,
        pretrained_backbone="./checkpoints_conv_hypernet/checkpoint.pth",
        pretrained_hypernet="./checkpoints_conv_hypernet/checkpoint.pth",
        resize=(256, 256))