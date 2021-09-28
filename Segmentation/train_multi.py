import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

from model.coeffnet.coeffnet import Multinet
from dataset.multiobj_dataset import Multiobj_Dataloader
from utils.pos_weight import get_pos_weight_from_batch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path = "/orion/u/pancy/data/object-pursuit/ithor/FloorPlan2"
prefix = "data_FloorPlan2_"

dataloader, dataset = Multiobj_Dataloader(data_dir=data_path, batch_size=16, num_workers=8, prefix=prefix, resize=(256, 256))
obj_num = dataset.obj_num
net = Multinet(obj_num=obj_num, z_dim=100, freeze_backbone=True).cuda()
net =  nn.DataParallel(net, device_ids=[0])

optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=4e-6, weight_decay=1e-7, momentum=0.9)

criterion = nn.BCEWithLogitsLoss()
# scheduler_lr=optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, mode='min', patience=10)

epochs = 200
checkpoints_path = './checkpoints_conv_small_full_simclr'
if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)
log_writer = open(os.path.join(checkpoints_path, "log.txt"), "w")

# log
log_writer.write(f"obj num: {dataset.obj_num} \n"
                f"object ident mapping: {dataset.classes} \n"
                f"checkpoints dir: {checkpoints_path} \n"
                f"epochs: {epochs} \n"
                f"data path: {data_path} \n"
                # f"cuda: {cuda} \n"
                f"parameter number of the network: {sum(x.numel() for x in net.parameters() if x.requires_grad)}\n")


for epoch in range(epochs):
    net.train()
    step = 0
    n_size = len(dataloader)
    loss_rec = []
    obj_loss_rec = []
    checkpoint_rec = []
    min_loss = 10
    obj_step = 0
    optimizer.zero_grad()
    with tqdm(total=n_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in dataloader:
            imgs = batch['image']
            true_masks = batch['mask']
            ident = batch['cls']
            
            imgs = imgs.to(dtype=torch.float32).cuda()
            mask_type = torch.float32
            true_masks = true_masks.to(dtype=mask_type).cuda()
            
            masks_pred, z = net(imgs, ident)
            # loss = criterion(masks_pred, true_masks)
            pos_weight = get_pos_weight_from_batch(true_masks)
            seg_loss = F.binary_cross_entropy_with_logits(masks_pred, true_masks, pos_weight=torch.tensor([pos_weight]).cuda())
            l1_loss = 0.1 * F.l1_loss(z, torch.zeros(z.size()).to(z.device))
            loss = seg_loss + l1_loss
            
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            loss_rec.append(loss.item())
            obj_loss_rec.append(loss.item())
            checkpoint_rec.append(loss.item())
            
            obj_step += 1
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            
            if obj_step == 1:
                # scheduler_lr.step(sum(obj_loss_rec)/len(obj_loss_rec))
                optimizer.step()
                obj_step = 0
                obj_loss_rec = []
                optimizer.zero_grad()
            
            pbar.update(1)
            step += 1
            
            if step % 100 == 0:
                avg_loss = sum(loss_rec)/len(loss_rec)
                log_writer.write(f"epoch {epoch}, step {step}, avg loss {avg_loss}\n")
                log_writer.flush()
                loss_rec = []
                
            if step % (n_size) == 0:
                if sum(checkpoint_rec)/len(checkpoint_rec) < min_loss:
                    min_loss = sum(checkpoint_rec)/len(checkpoint_rec)
                    torch.save(net.module.state_dict(), os.path.join(checkpoints_path, f'checkpoint.pth'))
                    log_writer.write(f"checkpoint saved ! \n")
                checkpoint_rec = []
                log_writer.flush()
                
log_writer.close()