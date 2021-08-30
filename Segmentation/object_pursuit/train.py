import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import optim

from model.coeffnet.coeffnet_simple import Singlenet, Coeffnet
from loss.dice_loss import dice_coeff
from loss.memory_loss import MemoryLoss

def set_eval(primary_net, hypernet, backbone=None):
    primary_net.eval()
    hypernet.eval()
    if backbone is not None:
        backbone.eval()
        
def set_train(primary_net, hypernet, backbone=None):
    primary_net.train()
    hypernet.train()
    if backbone is not None:
        backbone.train()

def eval_net(net_type, primary_net, loader, device, hypernet, backbone=None, zs=None):
    """Evaluation without the densecrf with the dice coefficient"""
    # set eval
    set_eval(primary_net, hypernet, backbone)

    n_val = len(loader)  # the number of batch
    tot = 0
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            # predict mask
            with torch.no_grad():
                if net_type == "singlenet":
                    mask_pred = primary_net(imgs, hypernet, backbone)
                elif net_type == "coeffnet":
                    assert zs is not None
                    mask_pred = primary_net(imgs, zs, hypernet, backbone)
                else:
                    raise NotImplementedError

            # cal dice coeff
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            res = dice_coeff(pred, true_masks).item()
            tot += res
            
            pbar.update()

    # set train
    set_train(primary_net, hypernet, backbone)
    
    return tot / n_val


def train_net(z_dim, 
              base_num, 
              dataset, 
              device, 
              net_type, 
              hypernet, 
              backbone=None, 
              zs=None, 
              save_cp_path=None, 
              base_dir=None, 
              epochs=20, 
              batch_size=16, 
              lr=0.0004, 
              val_percent=0.1):
    # set network
    if net_type == "singlenet":
        primary_net = Singlenet(z_dim)
    elif net_type == "coeffnet":
        assert zs is not None and len(zs) == base_num
        primary_net = Coeffnet(base_num, nn_init=True)
    else:
        raise NotImplementedError    
    primary_net.to(device)
    
    # set dataset and dataloader
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    # set checkpoint path
    if save_cp_path is not None and isinstance(save_cp_path, str):
        if not os.path.exists(save_cp_path):
            os.mkdir(save_cp_path)
    
    # optimize
    if backbone is not None:
        optim_param = filter(lambda p: p.requires_grad, itertools.chain(primary_net.parameters(), hypernet.parameters(), backbone.parameters()))
    else:
        optim_param = filter(lambda p: p.requires_grad, itertools.chain(primary_net.parameters(), hypernet.parameters()))
    optimizer = optim.RMSprop(optim_param, lr=lr, weight_decay=1e-7, momentum=0.9)
    
    SegLoss = nn.BCEWithLogitsLoss()
    if net_type == "singlenet":
        MemLoss = MemoryLoss(Base_dir=base_dir, device=device)
        mem_coeff = 0.01
        
    global_step = 0
    max_valid_acc = 0
        
    # training process
    for epoch in range(epochs):
        set_train(primary_net, hypernet, backbone)
        val_list = []
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                if net_type == "singlenet":
                    masks_pred = primary_net(imgs, hypernet, backbone)
                elif net_type == "coeffnet":
                    masks_pred = primary_net(imgs, zs, hypernet, backbone)
                else:
                    raise NotImplementedError
                
                seg_loss = SegLoss(masks_pred, true_masks)
                
                # optimize
                optimizer.zero_grad()
                seg_loss.backward()
                if net_type == "singlenet":
                    MemLoss(hypernet, mem_coeff)
                nn.utils.clip_grad_value_(optim_param, 0.1)
                optimizer.step()
                
                pbar.update(imgs.shape[0])
                global_step = 0
                
                # eval
                if global_step % int(n_train / (10*batch_size)) == 0:
                    val_score = eval_net(net_type, primary_net, val_loader, device, hypernet, backbone, zs)
                    val_list.append(val_score)
                    print('Validation Dice Coeff: {}'.format(val_score))
                    
        if save_cp_path is not None:
            avg_valid_acc = sum(val_list)/len(val_list)
            if avg_valid_acc > max_valid_acc:
                # TODO: save checkpoints
                max_valid_acc = avg_valid_acc
            
                
                
                