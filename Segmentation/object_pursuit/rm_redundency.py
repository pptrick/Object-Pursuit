import os
import torch
import json
from tqdm import tqdm
from model.coeffnet.hypernet import Hypernet
from model.coeffnet.coeffnet_simple import Backbone
from model.coeffnet.coeffnet_simple import init_backbone, init_hypernet
from object_pursuit.pursuit import get_z_bases, freeze
from object_pursuit.train import train_net
from dataset.basic_dataset import BasicDataset
from utils.util import *

def simplify_bases(log_dir, output_dir, base_path, record_path, hypernet_path, backbone_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), z_dim=100, resize=(256, 256), threshold=0.7):
    create_dir(log_dir)
    create_dir(output_dir)
    # init hypernet and backbone
    hypernet = Hypernet(z_dim)
    init_hypernet(hypernet_path, hypernet, device, freeze=True)
    hypernet.to(device)
    backbone = Backbone()
    init_backbone(backbone_path, backbone, device, freeze=True)
    backbone.to(device)
    # load initial bases
    init_bases = get_z_bases(z_dim, base_path, device)
    # load base info
    with open(record_path, 'r') as f:
        base_info = json.load(f)
    base_info = sorted(base_info, key=lambda e:e['acc'])
    for inf in base_info:
        inf["valid"] = True
    # start remove redundancy from low acc to high
    for obj in base_info:
        print(f"=============================start to run {obj['obj_name']}=======================================")
        if obj['acc'] <= threshold:
            obj['valid'] = False
            print(f"for obj {obj['obj_name']}, valid: {obj['valid']}")
        else:
            img_dir = os.path.join(obj['data_dir'], "imgs")
            mask_dir = os.path.join(obj['data_dir'], "masks")
            dataset = BasicDataset(img_dir, mask_dir, resize)
            freeze(hypernet=hypernet, backbone=backbone)
            cp_dir = os.path.join(log_dir, str(obj["index"])+"_"+obj["obj_name"])
            create_dir(cp_dir)
            zs = []
            for inf in base_info:
                if inf['valid'] and inf['index'] != obj['index']:
                    zs.append(init_bases[inf['index']])
            max_val_acc, _ = train_net(z_dim=z_dim, base_num=len(zs), dataset=dataset, device=device,
                      zs=zs, 
                      net_type="coeffnet", 
                      hypernet=hypernet, 
                      backbone=backbone,
                      save_cp_path=cp_dir,
                      batch_size=64,
                      max_epochs=200,
                      lr=4e-4,
                      acc_threshold=threshold)
            if max_val_acc > threshold:
                obj['valid'] = False
            print(f"for obj {obj['obj_name']}, current base num {len(zs)}, max acc {max_val_acc}, valid: {obj['valid']}")
    # save
    new_bases = []
    new_base_info = []
    count = 0
    for obj in base_info:
        if obj['valid']:
            new_bases.append(init_bases[obj['index']])
            obj['index'] = count
            obj['base_file'] = f"base_{'%04d' % count}.json"
            new_base_info.append(obj)
            count += 1
    for i in range(len(new_bases)):
        saved_file_path = os.path.join(output_dir, f"base_{'%04d' % i}.json")
        torch.save({'z':new_bases[i]}, saved_file_path)
    with open(os.path.join(log_dir, "base_info.json"), 'w') as f:
        json.dump(new_base_info, f)
    
    