"""
This script contains tools for evaluating hypernet and bases aquired from the pursuit process
"""
import os
import json
import torch
from torch.utils.data import DataLoader

from dataset.basic_dataset import BasicDataset
from model.coeffnet.coeffnet import Singlenet
from evaluation.eval_net import eval_net
    
def getObjDataPath(dataset, data_dir, obj, res="480p"):
    if dataset == "DAVIS":
        img_ent = "JPEGImages"
        mask_ent = "Annotations"
        img_dir = os.path.join(data_dir, img_ent, res, obj)
        mask_dir = os.path.join(data_dir, mask_ent, res, obj)
        if (not os.path.isdir(img_dir)) or (not os.path.isdir(mask_dir)):
            print(f"[Warning] {img_dir} or {mask_dir} may not be correct directories")
        return img_dir, mask_dir
    else:
        raise NotImplementedError
    
def evalPursuit(z_dim, device, dataset, data_dir, ckpt_dir, batch_size=8, use_backbone=False):
    assert os.path.isdir(ckpt_dir)
    with open(os.path.join(ckpt_dir, "z_info.json"), 'r') as f:
        z_info = json.load(f)
    record_acc = []
    for z_inf in z_info:
        z_file = os.path.join(ckpt_dir, "zs", z_inf["z_file"])
        if dataset == "DAVIS":
            img_dir, mask_dir = getObjDataPath("DAVIS", data_dir, z_inf["data_dir"])
            val_objects = ["blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl", "dog", "drift-chicane", "drift-straight", "goat", "horsejump-high", "kite-surf", "libby", "motocross-jump", "paragliding-launch", "parkour", "scooter-black", "soapbox"]
        Dataset = BasicDataset(img_dir, mask_dir, resize=(256, 256), random_crop=True)
        net = Singlenet(z_dim, device, use_backbone=use_backbone)
        net.load_z(z_file)
        net.init_hypernet(os.path.join(ckpt_dir, "checkpoint", "hypernet.pth"))
        net.to(device)
        loader = DataLoader(Dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        acc, _ = eval_net(net, loader, device, use_IOU=True)
        print(f"[Pursuit Evaluation] eval acc for obj {z_inf['data_dir']} is {acc}")
        if z_inf['data_dir'] in val_objects:
            record_acc.append(acc)
            print(f"save {z_inf['data_dir']}'s acc")
    print(f"avg acc for val obj is {sum(record_acc)/len(record_acc)}, val obj num: {len(record_acc)}")




      