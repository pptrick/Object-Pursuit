import os
import torch
import random
import json
from model.coeffnet.coeffnet import Singlenet
from dataset.basic_dataset import BasicDataset
from torch.utils.data import DataLoader, sampler
from evaluation.eval_net import eval_net
from utils.util import *

def test_unit(data_dir, z_dir, hypernet_path, backbone_path, test_size=400, start_index=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Singlenet(z_dim=100, device=device)
    net.to(device=device)
    net.init_hypernet(hypernet_path)
    net.init_backbone(backbone_path)
    
    z_files = [os.path.join(z_dir, zf) for zf in sorted(os.listdir(z_dir)) if zf.endswith('.json')]
    img_dir = os.path.join(data_dir, "imgs")
    mask_dir = os.path.join(data_dir, "masks")
    eval_dataset = BasicDataset(img_dir, mask_dir, (256, 256))
    
    n_size = len(eval_dataset)
    indices = [i for i in range(n_size)]
    random.shuffle(indices)
    if test_size is not None:
        eval_sampler = sampler.SubsetRandomSampler(indices[:min(n_size, test_size)])
    else:
        eval_sampler = sampler.SubsetRandomSampler(indices[:n_size])
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, drop_last=True, sampler=eval_sampler)
    
    max_acc = 0.0
    argmax_zf = None
    argmax_index = None
    for i in range(start_index, len(z_files)):
        zf = z_files[i]
        net.load_z(zf)
        res, _ = eval_net(net, eval_loader, device)
        print(f"z file: {zf}, test accuracy: {res}")
        if res >= max_acc:
            max_acc = res
            argmax_zf = zf
            argmax_index = i
    return max_acc, argmax_index, argmax_zf
    
def test_seen_obj(obj_info, z_info, z_dir, hypernet_path, backbone_path, log_name, threshold=0.7):
    log_file = open(log_name, 'w')
    max_threshold = 0.8
    assert os.path.isfile(obj_info) and obj_info.endswith(".json")
    total_count = 0
    seen_count = 0
    unseen_count = 0
    correct_count = 0
    false_count = 0
    with open(obj_info, 'r') as f:
        obj_info = json.load(f)
        for obj in obj_info:
            if obj["acc"] > max_threshold:
                data_dir = obj["data_dir"]
                obj_name = obj["obj_name"]
                write_log(log_file, f"Start testing object {data_dir}, obj name: {obj_name}...")
                test_acc, test_index, test_zf = test_unit(data_dir, z_dir, hypernet_path, backbone_path)
                write_log(log_file, f"The argmax z file is {test_zf}, index: {test_index}, max acc: {test_acc}")
                total_count += 1
                if test_acc > threshold:
                    seen_count += 1
                    with open(z_info, 'r') as z_inf:
                        Z_info = json.load(z_inf)
                        for zi in Z_info:
                            if zi["obj_name"] == obj_name:
                                index = zi["index"]
                        if index == test_index:
                            write_log(log_file, "correct !")
                            correct_count += 1
                        else:
                            write_log(log_file, "false !")
                            false_count += 1
                else:
                    unseen_count += 1
    res_info = f'''test on seen objects:
        total count:                            {total_count}
        seen count:                             {seen_count}
        unseen count:                           {unseen_count}
        correct count(in seen count):           {correct_count}
        false count(in seen count):             {false_count}
    '''
    write_log(log_file, res_info)
    log_file.close()
            
def test_unseen_obj(obj_dir, z_dir, hypernet_path, backbone_path, log_name, threshold=0.7):
    log_file = open(log_name, 'w')
    assert os.path.isdir(obj_dir) and os.path.isdir(z_dir)
    data_dirs = [os.path.join(obj_dir, d) for d in sorted(os.listdir(obj_dir))]
    total_count = 0
    seen_count = 0
    unseen_count = 0
    for data_dir in data_dirs:
        write_log(log_file, f"Start testing object {data_dir}...")
        test_acc, test_index, test_zf = test_unit(data_dir, z_dir, hypernet_path, backbone_path, start_index=31)
        total_count += 1
        if test_acc > threshold:
            seen_count += 1
            write_log(log_file, f"find a seen object! test acc {test_acc}, index {test_index}, test z file {test_zf}")
        else:
            unseen_count += 1
            write_log(log_file, f"find a unseen object! test acc {test_acc}, index {test_index}, test z file {test_zf}")
    res_info = f'''test on seen objects:
        total count:                            {total_count}
        seen count:                             {seen_count}
        unseen count:                           {unseen_count}
    '''
    write_log(log_file, res_info)
    log_file.close()
    
        