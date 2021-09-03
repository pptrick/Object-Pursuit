import torch
from object_pursuit.pursuit import pursuit

pursuit(z_dim=100, 
        data_dir="/data/pancy/iThor/single_obj/FloorPlan2",
        output_dir="./checkpoints_objectpursuit_test",
        device=torch.device('cuda:5' if torch.cuda.is_available() else 'cpu'),
        pretrained_bases=None,
        pretrained_backbone="./checkpoints_conv_small/checkpoint.pth",
        pretrained_hypernet="./checkpoints_conv_small/checkpoint.pth",
        resize=(256, 256),
        select_strat="random")