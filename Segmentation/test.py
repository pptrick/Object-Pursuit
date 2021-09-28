import os

from evaluation.seen_obj import test_seen_obj, test_unseen_obj
from utils.util import *

if __name__ == "__main__":
    thres = 0.6
    rd = 48
    test_dir = f"./checkpoints_test_{thres}_round_{rd}"
    create_dir(test_dir)
    
    test_unseen_obj(obj_dir="/orion/u/pancy/data/object-pursuit/ithor/Dataset/Test",
                    z_dir=f'./checkpoints_sequence_threshold_{thres}/checkpoint/checkpoint_round_{rd}/zs/',
                    hypernet_path=f'./checkpoints_sequence_threshold_{thres}/checkpoint/checkpoint_round_{rd}/hypernet.pth',
                    backbone_path=f'./checkpoints_sequence_threshold_{thres}/checkpoint/backbone.pth',
                    log_name=os.path.join(test_dir, "test_unseen_log.txt"),
                    threshold=thres)
    
    # test_seen_obj(obj_info='./checkpoints_rm_redundancy_0.9/base_info.json',
    #               z_info='./checkpoints_below_acc/base_info.json',
    #               z_dir=f'./checkpoints_sequence_threshold_{thres}/checkpoint/checkpoint_round_{rd}/zs/', 
    #               hypernet_path=f'./checkpoints_sequence_threshold_{thres}/checkpoint/checkpoint_round_{rd}/hypernet.pth',
    #               backbone_path=f'./checkpoints_sequence_threshold_{thres}/checkpoint/backbone.pth',
    #               log_name=os.path.join(test_dir, "test_seen_log.txt"),
    #               threshold=thres)