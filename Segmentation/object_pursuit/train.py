import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from tqdm import tqdm, utils
from torch.utils.data import DataLoader, random_split
from torch import optim

from model.coeffnet.coeffnet_simple import Singlenet, Coeffnet
from loss.dice_loss import dice_coeff
from loss.IoU_loss import IoULoss
from loss.memory_loss import MemoryLoss
from utils.pos_weight import get_pos_weight_from_batch
from utils.util import *

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
    set_eval(primary_net, hypernet)

    n_val = len(loader)  # the number of batch
    tot = 0
    
    # in case there's only one batch
    if n_val == 0:
        n_val = 1
    
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
              z_dir=None, 
              max_epochs=80, 
              batch_size=64, 
              lr=0.0004, 
              val_percent=0.1,
              wait_epochs=3,
              acc_threshold=0.90,
              l1_loss_coeff=0.2,
              mem_loss_coeff=0.04):
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
    maximum_len = 2500
    if len(dataset) > maximum_len:
        n_data = maximum_len
    else:
        n_data = len(dataset)
    n_val = int(n_data * val_percent)
    n_train = int(n_data * (1-val_percent))
    n_rest = len(dataset) - n_val - n_train
    train, val, _ = random_split(dataset, [n_train, n_val, n_rest])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    # n_train = len(dataset)
    # n_val = len(dataset)
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    # val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    # optimize
    if backbone is not None:
        optim_param = filter(lambda p: p.requires_grad, itertools.chain(primary_net.parameters(), hypernet.parameters(), backbone.parameters()))
    else:
        optim_param = filter(lambda p: p.requires_grad, itertools.chain(primary_net.parameters(), hypernet.parameters()))
    optimizer = optim.RMSprop(optim_param, lr=lr, weight_decay=1e-7, momentum=0.9)
    
    if net_type == "singlenet":
        MemLoss = MemoryLoss(Base_dir=z_dir, device=device)
        mem_coeff = mem_loss_coeff
        
    global_step = 0
    max_valid_acc = 0
    max_record = None
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
        z_dir:           {z_dir}
        wait epochs:     {wait_epochs}
        val acc thres:   {acc_threshold}
        trainable parameter number of the primarynet: {sum(x.numel() for x in primary_net.parameters() if x.requires_grad)}
        trainable parameter number of the hypernet: {sum(x.numel() for x in hypernet.parameters() if x.requires_grad)}
        trainable parameter number of the backbone: {sum(x.numel() for x in backbone.parameters() if x.requires_grad)}
    '''
    write_log(log_file, info_text)
        
    # training process
    try:
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
                    
                    seg_loss = F.binary_cross_entropy_with_logits(masks_pred, true_masks, pos_weight=torch.tensor([get_pos_weight_from_batch(true_masks)]).to(device))
                    regular_loss = primary_net.L1_loss(l1_loss_coeff)
                    loss = seg_loss + regular_loss
                    pbar.set_postfix(**{'seg loss (batch)': loss.item()})
                    
                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    if net_type == "singlenet":
                        MemLoss(hypernet, mem_coeff)
                    nn.utils.clip_grad_value_(optim_param, 0.1)
                    optimizer.step()
                    
                    pbar.update(imgs.shape[0])
                    global_step += 1
                    
                    # eval
                    if global_step % int(n_train / (2*batch_size)) == 0:
                        val_score = eval_net(net_type, primary_net, val_loader, device, hypernet, backbone, zs)
                        val_list.append(val_score)
                        write_log(log_file, f'  Validation Dice Coeff: {val_score}, segmentation loss + l1 loss: {loss}')
                        
            if save_cp_path is not None:
                if len(val_list) > 0:
                    avg_valid_acc = sum(val_list)/len(val_list)
                    if avg_valid_acc > max_valid_acc:
                        if net_type == "singlenet":
                            max_record = primary_net.z
                            torch.save(primary_net.state_dict(), os.path.join(save_cp_path, f'Best_z.pth'))
                        elif net_type == "coeffnet":
                            max_record = primary_net.coeffs
                            torch.save(primary_net.state_dict(), os.path.join(save_cp_path, f'Best_coeff.pth'))
                        max_valid_acc = avg_valid_acc
                        stop_counter = 0
                        write_log(log_file, f'epoch {epoch} checkpoint saved! best validation acc: {max_valid_acc}')
                    else:
                        stop_counter += 1
                    
                    if stop_counter >= wait_epochs or max_valid_acc > acc_threshold:
                        # stop procedure
                        write_log(log_file, f'training stopped at epoch {epoch}')
                        write_log(log_file, f'current record value (coeff or z): {max_record}')
                        log_file.close()
                        return max_valid_acc, primary_net
    except Exception as e:
        write_log(log_file, f'Error catch during training! info: {e}')
        return 0.0, primary_net
    
    #stop procedure
    write_log(log_file, f'training stopped')
    write_log(log_file, f'current record value (coeff or z): {max_record}')
    log_file.close()
    return max_valid_acc, primary_net
            

def have_seen(dataset, device, z_dir, z_dim, hypernet, backbone, threshold, start_index=0, test_percent=0.2, batch_size=64):
    primary_net = Singlenet(z_dim)
    primary_net.to(device)
    
    n_test = int(len(dataset)*test_percent)
    n_rest = len(dataset) - n_test
    test_set, _ = random_split(dataset, [n_test, n_rest])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    z_files = [os.path.join(z_dir, zf) for zf in sorted(os.listdir(z_dir)) if zf.endswith('.json')]
    max_acc = 0.0
    max_zf = None
    count = 0
    for zf in z_files:
        if count < start_index:
            count += 1
            continue
        primary_net.load_z(zf)
        test_acc = eval_net(net_type="singlenet", primary_net=primary_net, loader=test_loader, device=device, hypernet=hypernet, backbone=backbone)
        if test_acc > max_acc:
            max_acc = test_acc
            max_zf = zf
        count += 1
    if max_acc > threshold:
        return True, max_acc, max_zf
    else:
        return False, max_acc, max_zf
            
    
         