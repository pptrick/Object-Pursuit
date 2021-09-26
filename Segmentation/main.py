import torch
from utils.GenBases import genBases
from object_pursuit.pursuit import pursuit
from object_pursuit.rm_redundency import simplify_bases

# genBases('./checkpoints_conv_small_full/checkpoint.pth', './Bases/', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

pursuit(z_dim=100, 
        data_dir="/orion/u/pancy/data/object-pursuit/ithor/Dataset/Train",
        output_dir="./checkpoints_random_1_threshold_0.7",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        pretrained_bases="./Bases_0.9/",
        pretrained_backbone="./checkpoints_conv_small_full/checkpoint.pth",
        pretrained_hypernet="./checkpoints_conv_small_full/checkpoint.pth",
        resize=(256, 256),
        select_strat="random",
        express_threshold=0.7,
        log_info="Data: sequence; threshold: 0.7")

# simplify_bases(log_dir='./checkpoints_rm_redundancy_0.85/',
#                output_dir='./Bases_0.85/',
#                base_path="./checkpoints_conv_small_full/checkpoint.pth",
#                hypernet_path="./checkpoints_conv_small_full/checkpoint.pth",
#                backbone_path="./checkpoints_conv_small_full/checkpoint.pth",
#                record_path="./checkpoints_conv_small_full/record.json")