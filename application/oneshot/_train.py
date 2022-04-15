import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.eval_net import eval_net

from dataset.visualize import vis_predict

from utils.util import create_dir, write_log

def train_nshot(net,
                device,
                train_dataset, # torch dataset
                eval_dataset,
                epochs=10,
                batch_size=8,
                lr=4e-4,
                ckpt_path="./checkpoint_nshot/",
                eval_step=10, # evaluate every 'eval_step' steps
                save_ckpt=False, # save checkpoint (model param)
                save_viz=False, # save visualization results
                use_dice=False,
                args=None):
    # dataset
    n_train = len(train_dataset)
    n_val = len(eval_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    # log
    create_dir(ckpt_path)
    logf = open(os.path.join(ckpt_path, f"log_{time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime())}.txt"), "w")
    info_text = f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ckpt_path}
        Device:          {device}
        Eval step:       {eval_step}
        save checkpoint: {save_ckpt}
        save visualize:  {save_viz}
        use dice loss:   {use_dice}
        parameter number of the network: {sum(x.numel() for x in net.parameters() if x.requires_grad)}
    \n'''
    write_log(logf, info_text)
    
    if args is not None:
        data_text = f'''Data & Model info:
            model:           {args.model}
            dataset:         {args.dataset}
            n (n-shot):      {args.n}
            z dim:           {args.z_dim}
            img dir:         {args.img_dir}
            mask dir:        {args.mask_dir}
            bases dir:       {args.bases_dir}
            backbone:        {args.pretrained_backbone}
            hypernet:        {args.pretrained_hypernet}
            resize:          {args.resize}
        \n'''
    write_log(logf, data_text)
    
    # training settings
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-7, momentum=0.9)
    global_step = 0
    max_val_acc = 0
    
    # start training
    for epoch in range(epochs):
        write_log(logf, f"Start epoch {epoch} ...")
        net.train()
        val_acc_list = [] # record validation accuracy in one epoch
        loss_list = [] # record loss in one epoch
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                masks = batch['mask']
                
                assert imgs.shape[1] == 3 # deal with rgb image only (for now)
                
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.float32) # torch.float32 for single object seg (n_class=1), else should be torch.long
                
                # forward
                pred = net(imgs)
                # backward
                loss = F.binary_cross_entropy_with_logits(pred, masks)
                loss_list.append(loss.item())
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                
                # update
                pbar.update(imgs.shape[0])
                global_step += 1
                
                # eval
                if global_step % int(eval_step * int(n_train / (batch_size))) == 0:
                    val_score, d = eval_net(net, val_loader, device, use_IOU=(not use_dice))
                    val_acc_list.append(val_score)
                    write_log(logf, f'Validation Dice Coeff: {val_score}, decay: {d[0]}, current loss: {sum(loss_list)/len(loss_list)}')
                    loss_list = []
                    
        # An epoch finished & save ckpt
        if len(val_acc_list) > 0: # eval acc has been recorded
            avg_val_acc = sum(val_acc_list)/len(val_acc_list)
            if avg_val_acc > max_val_acc:
                write_log(logf, f'Find new max validation acc: {avg_val_acc} at epoch {epoch}')
                if save_ckpt:
                    save_path = os.path.join(ckpt_path, f'ckpt_best.pth')
                    torch.save(net.state_dict(), save_path)
                    write_log(logf, f'save checkpoint to {save_path}')
                if save_viz:
                    save_path = os.path.join(ckpt_path, 'viz_pred')
                    vis_predict(save_path, net, val_loader, device)
                    write_log(logf, f'save visualization predict result to {save_path}')
                max_val_acc = avg_val_acc
    
    write_log(logf, "Training ends!") 
    logf.close()    

