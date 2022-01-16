import os
import time
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

from pretrain._eval import joint_eval

from utils.pos_weight import get_pos_weight_from_batch
from utils.util import create_dir, write_log

def joint_train(net,
                device,
                dataloader_train,
                dataset_eval,
                epochs=100,
                batch_size=8,
                lr=1e-4,
                ckpt_path="./checkpoints_pretrain/",
                eval_step=10,
                n_val=-1,
                save_ckpt=True,
                use_dice=False,
                args=None):
    
    # init
    n_size = len(dataloader_train)
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-7, momentum=0.9)
    scheduler_lr=optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # log & checkpoint
    create_dir(ckpt_path)
    logf = open(os.path.join(ckpt_path, f"log_{time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime())}.txt"), "w")
    if args is not None:
        data_text = f'''Data & Model info:
            dataset:         {args.dataset}
            z dim:           {args.z_dim}
            data dir:        {args.data_dir}
            resize:          {args.resize}
            num balance:     {args.num_balance}
            use backbone:    {args.use_backbone}
            freeze backbone: {args.freeze_backbone}
        \n'''
        write_log(logf, data_text)
    
    info_text = f'''Starting training:
        Epochs:          {epochs}
        data size:       {n_size}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {ckpt_path}
        Device:          {device}
        Eval step:       {eval_step}
        Eval data num:   {n_val}
        save checkpoint: {save_ckpt}
        use dice loss:   {use_dice}
        parameter number of the network: {sum(x.numel() for x in net.parameters() if x.requires_grad)}
    \n'''
    write_log(logf, info_text)
    
    # recorder
    max_eval_acc = 0
    
    for epoch in range(epochs):
        net.train()
        loss_recorder = []
        write_log(logf, f"***********epoch {epoch} started**********")
        with tqdm(total=n_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in dataloader_train:
                imgs = batch['image']
                true_masks = batch['mask']
                ident = batch['cls'][0].item() # assume that object class in one batch are same
                
                assert imgs.shape[1] == 3 # deal with rgb image only (for now)
                
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                # forward
                masks_pred, z = net(imgs, ident)
                loss = F.binary_cross_entropy_with_logits(masks_pred, true_masks)
                
                # pos_weight = get_pos_weight_from_batch(true_masks)
                # seg_loss = F.binary_cross_entropy_with_logits(masks_pred, true_masks, pos_weight=torch.tensor([pos_weight]).cuda())
                # l1_loss = 0.1 * F.l1_loss(z, torch.zeros(z.size()).to(z.device))
                # loss = seg_loss + l1_loss
                
                # backward
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                loss_recorder.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(1)
                
        scheduler_lr.step()
        
        # eval & save
        write_log(logf, f"Specific loss list: {loss_recorder} \n")  
        write_log(logf, f"epoch {epoch} ended, mean loss {sum(loss_recorder)/len(loss_recorder)} \n")
        
        if (epoch+1) % eval_step == 0:
            write_log(logf, f"Start Joint Evaluation...") 
            eval_ckpt_path = os.path.join(ckpt_path, "eval")
            acc = joint_eval(net, dataset_eval, device, epoch, eval_ckpt_path, n_val, batch_size, use_IOU=(not use_dice))
            write_log(logf, f"Joint Evaluation: mean acc {acc}, check {eval_ckpt_path} for details!") 
            if acc > max_eval_acc:
                max_eval_acc = acc
                write_log(logf, f"mean acc {acc} reached the highest eval acc ~") 
                if save_ckpt:
                    torch.save(net.state_dict(), os.path.join(ckpt_path, f'pretrain_epoch_{epoch}.pth'))
                    write_log(logf, f"checkpoint saved") 
        
    write_log(logf, "Training ends!") 
    logf.close()  