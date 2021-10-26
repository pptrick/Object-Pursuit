import torch
import argparse
from utils.GenBases import genBases
from object_pursuit.pursuit import pursuit
from object_pursuit.rm_redundency import simplify_bases

# genBases('./checkpoints_conv_small_full/checkpoint.pth', './Bases/', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--rand-num', dest='rand_num', type=int, nargs='?', default=5,
                        help='random number')

    return parser.parse_args()


args = get_args()
chuanyu_dir = "/orion/u/pancy/project/Object-Pursuit/Segmentation"

pursuit(z_dim=100, 
        data_dir="/orion/u/pancy/data/object-pursuit/ithor/Dataset/Train",
        output_dir=f"./checkpoints_pursuit_allweights_0.8",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        initial_zs=f"{chuanyu_dir}/Bases_allweights/",
        pretrained_bases=f"{chuanyu_dir}/Bases_allweights/",
        pretrained_backbone=f"{chuanyu_dir}/checkpoints_conv_allweights/checkpoint.pth",
        pretrained_hypernet=f"{chuanyu_dir}/checkpoints_conv_allweights/checkpoint.pth",
        resize=(256, 256),
        select_strat="sequence",
        express_threshold=0.8,
        use_backbone=False,
        log_info="Data: sequence; threshold: 0.8")

# simplify_bases(log_dir='./checkpoints_simple_zs/',
#                output_dir='./Bases_allweights/',
#                base_path="./checkpoints_conv_allweights/checkpoint.pth",
#                hypernet_path="./checkpoints_conv_allweights/checkpoint.pth",
#                backbone_path="./checkpoints_conv_allweights/checkpoint.pth",
#                record_path="./checkpoints_conv_allweights/record.json",
#                z_dim=100,
#                threshold=0.7,
#                use_backbone=False)