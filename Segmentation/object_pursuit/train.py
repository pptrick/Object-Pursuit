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
        
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def write_log(log_file, string):
    print(string)
    log_file.write(string+'\n')
    log_file.flush()

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
              max_epochs=80, 
              batch_size=16, 
              lr=0.0004, 
              val_percent=0.1,
              wait_epochs=3,
              acc_threshold=0.95):
    # set logger
    log_file = open(os.path.join(save_cp_path, "log.txt"), "w")

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
    stop_counter = 0
    
    # write info
    info_text = f'''Starting training:
        net type:        {net_type}
        Max epochs:      {max_epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp_path}
        Device:          {device}
        base_dir:        {base_dir}
        wait epochs:     {wait_epochs}
        val acc thres:   {acc_threshold}
        trainable parameter number of the primarynet: {sum(x.numel() for x in primary_net.parameters() if x.requires_grad)}
        trainable parameter number of the hypernet: {sum(x.numel() for x in hypernet.parameters() if x.requires_grad)}
        trainable parameter number of the backbone: {sum(x.numel() for x in backbone.parameters() if x.requires_grad)}
    '''
    write_log(log_file, info_text)
        
    # training process
    for epoch in range(max_epochs):
        set_train(primary_net, hypernet, backbone)
        val_list = []
        write_log(log_file, f"Start epoch {epoch}")
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{max_epochs}', unit='img') as pbar:
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
                
                pbar.set_postfix(**{'seg loss (batch)': seg_loss.item()})
                
                # optimize
                optimizer.zero_grad()
                seg_loss.backward()
                if net_type == "singlenet":
                    MemLoss(hypernet, mem_coeff)
                nn.utils.clip_grad_value_(optim_param, 0.1)
                optimizer.step()
                
                pbar.update(imgs.shape[0])
                global_step += 1
                
                # eval
                if global_step % int(n_train / (10*batch_size)) == 0:
                    val_score = eval_net(net_type, primary_net, val_loader, device, hypernet, backbone, zs)
                    val_list.append(val_score)
                    write_log(log_file, f'  Validation Dice Coeff: {val_score}, segmentation loss: {seg_loss}')
                    
        if save_cp_path is not None:
            avg_valid_acc = sum(val_list)/len(val_list)
            if avg_valid_acc > max_valid_acc:
                if net_type == "singlenet":
                    torch.save(primary_net.state_dict(), os.path.join(save_cp_path, f'Best_z.pth'))
                    primary_net.save_z(os.path.join(base_dir, f'base_{base_num}.json'), hypernet)
                elif net_type == "coeffnet":
                    torch.save(primary_net.state_dict(), os.path.join(save_cp_path, f'Best_coeff.pth'))
                max_valid_acc = avg_valid_acc
                stop_counter = 0
                write_log(log_file, f'epoch {epoch} checkpoint saved! best validation acc: {max_valid_acc}')
            else:
                stop_counter += 1
        
            if stop_counter >= wait_epochs or max_valid_acc > acc_threshold:
                # stop procedure
                write_log(log_file, f'training stopped at epoch {epoch}')
                log_file.close()
                return max_valid_acc
    
    #stop procedure
    write_log(log_file, f'training stopped')
    log_file.close()
    return max_valid_acc
            
                
                
                