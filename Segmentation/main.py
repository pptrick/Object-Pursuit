import torch
from utils.GenBases import genBases
from object_pursuit.pursuit import pursuit
from object_pursuit.rm_redundency import simplify_bases

# genBases('./checkpoints_conv_small_full/checkpoint.pth', './Bases/', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

pursuit(z_dim=100, 
        data_dir="/orion/u/pancy/data/object-pursuit/ithor/Dataset/Train",
        output_dir="./checkpoints_alpha2_0.0",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        initial_zs="./Bases/",
        pretrained_bases="./Bases/",
        pretrained_backbone="./checkpoints_conv_small_full_frzbackbone/checkpoint.pth",
        pretrained_hypernet="./checkpoints_conv_small_full_frzbackbone/checkpoint.pth",
        resize=(256, 256),
        select_strat="sequence",
        express_threshold=0.6,
        log_info="Data: sequence; threshold: 0.6")

# simplify_bases(log_dir='./checkpoints_simple_zs/',
#                output_dir='./Bases/',
#                base_path="./checkpoints_conv_small_full_frzbackbone/checkpoint.pth",
#                hypernet_path="./checkpoints_conv_small_full_frzbackbone/checkpoint.pth",
#                backbone_path="./checkpoints_conv_small_full_frzbackbone/checkpoint.pth",
#                record_path="./checkpoints_conv_small_full_frzbackbone/record.json",
#                threshold=0.69)