import os
import torch
import torch.nn.functional as F
import time
import json
import random
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from loss.dice_loss import dice_coeff
from loss.criterion import jaccard

from pretrain._dataset import MultiJointDataset
from pretrain._model import Multinet

from utils.util import create_dir

def _eval(multinet, loader, ident, device, use_IOU=False):
    multinet.train()
    mask_type = torch.float32 if multinet.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred, _ = multinet(imgs, ident) # multinet takes img and ident as input

                if multinet.n_classes > 1:
                    res = F.cross_entropy(mask_pred, true_masks).item()
                    tot += res
                else:
                    pred = torch.sigmoid(mask_pred)
                    pred = (pred > 0.5).float()
                    if use_IOU:
                        res = jaccard(true_masks, pred)
                    else:
                        res = dice_coeff(pred, true_masks).item()
                    tot += res
                    
            pbar.update()

    multinet.train()
    
    return tot / n_val

def getDataloader(dataset, n_val, batch_size):
    n_size = len(dataset)
    indices = [i for i in range(n_size)]
    random.shuffle(indices)
    if n_val is not None and n_val >= 0:
        eval_sampler = sampler.SubsetRandomSampler(indices[:min(n_size, n_val)])
    else:
        eval_sampler = sampler.SubsetRandomSampler(indices[:n_size])
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True, sampler=eval_sampler)
    return eval_loader

def joint_eval(multinet, 
               multidataset, 
               device,
               epoch_num,
               eval_ckpt,
               n_val=-1,  
               batch_size=8,
               use_IOU=False):
    # multinet and multidataset should be type-specified
    assert isinstance(multidataset, MultiJointDataset)
    # ckpt & record
    eval_record = []
    mean_acc = []
    # set up dataset
    datasets = multidataset.datasets
    # start eval
    for ds in datasets:
        loader = getDataloader(ds["dataset"], n_val, batch_size)
        eval_acc = _eval(multinet, loader, ds["index"], device, use_IOU)
        eval_record.append({
            "index": ds["index"],
            "img_dir": ds["img_dir"],
            "mask_dir": ds["mask_dir"],
            "eval_len": batch_size * len(loader),
            "acc": eval_acc
        })
        mean_acc.append(eval_acc)
        print(f"Object {ds['index']} tested, mean acc {eval_acc}, data_dir: {ds['img_dir']}")
    if len(mean_acc) <= 0:
        mean_acc = 0
    else:
        mean_acc = sum(mean_acc)/len(mean_acc)
        
    # save ckpt
    create_dir(eval_ckpt)
    with open(os.path.join(eval_ckpt, f"log_epoch_{'%04d' % epoch_num}_acc_{mean_acc}_time_{time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime())}.json"), 'w') as f:
        json.dump(eval_record, f)
        
    return mean_acc
        
        