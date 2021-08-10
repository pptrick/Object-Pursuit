import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from model.coeffnet.coeffnet import Multinet
from utils.multiobj_dataset import Multiobj_Dataloader

data_path = "/data/pancy/iThor/single_obj/FloorPlan2"
prefix = "data_FloorPlan2_"
cuda = 2

device = torch.device('cuda:'+str(cuda) if torch.cuda.is_available() else 'cpu')
dataloader, dataset = Multiobj_Dataloader(data_dir=data_path, batch_size=16, num_workers=8, prefix=prefix)
net = Multinet(obj_num=dataset.obj_num, z_dim=100, device=device)
net.to(device=device)

optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0004, weight_decay=1e-7, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

epochs = 100
checkpoints_path = './checkpoints_equal'
if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)
log_writer = open(os.path.join(checkpoints_path, "log.txt"), "w")

# log
log_writer.write(f"obj num: {dataset.obj_num} \n"
                f"object ident mapping: {dataset.classes} \n"
                f"checkpoints dir: {checkpoints_path} \n"
                f"epochs: {epochs} \n"
                f"data path: {data_path} \n"
                f"cuda: {cuda} \n"
                f"parameter number of the network: {sum(x.numel() for x in net.parameters() if x.requires_grad)}\n")


for epoch in range(epochs):
    net.train()
    step = 0
    n_size = len(dataloader)
    loss_rec = []
    with tqdm(total=n_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in dataloader:
            imgs = batch['image']
            true_masks = batch['mask']
            ident = batch['cls']
            
            assert imgs.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'
                
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            
            masks_pred = net(imgs, ident)
            loss = criterion(masks_pred, true_masks)
            
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            loss_rec.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            
            pbar.update(1)
            step += 1
            
            if step % 100 == 0:
                avg_loss = sum(loss_rec)/len(loss_rec)
                log_writer.write(f"epoch {epoch}, step {step}, avg loss {avg_loss}\n")
                log_writer.flush()
                loss_rec = []
                
            if step % (n_size//5) == 0:
                torch.save(net.state_dict(), os.path.join(checkpoints_path, f'checkpoint.pth'))
                log_writer.write(f"checkpoint saved ! \n")
                log_writer.flush()
                
log_writer.close()