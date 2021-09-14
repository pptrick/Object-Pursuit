import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.modules import loss
from tqdm import tqdm

from eval import eval_net

from dataset.basic_dataset import BasicDataset
from dataset.davis_dataset import DavisDataset, OneshotDavisDataset
from dataset.kitti_dataset import KittiTrainDataset, KittiTestDataset
from dataset.visualize import vis_predict
from torch.utils.data import DataLoader, dataset, random_split

from model.deeplabv3.deeplab import *
from model.unet import UNet
from model.coeffnet.coeffnet import Coeffnet, Singlenet
from utils.pos_weight import get_pos_weight_from_batch

from loss.memory_loss import MemoryLoss
from loss.dice_loss import DiceCoeff

obj_train = 'cows'
obj = 'AlarmClock'
dir_img = [f'/orion/u/pancy/data/object-pursuit/ithor/FloorPlan2/data_FloorPlan2_{obj}/imgs']
dir_mask = [f'/orion/u/pancy/data/object-pursuit/ithor/FloorPlan2/data_FloorPlan2_{obj}/masks']
dir_checkpoint = f'checkpoints_kitti_4'

acc = []

def train_net(args,
              net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=False,
              img_scale=0.5,
              use_mem_loss=False):

    # dataset = BasicDataset(dir_img, dir_mask, resize=(256, 256), train=True)
    # train_percent = 0.9
    # val_percent = 0.1
    # n_val = int(len(dataset) * val_percent)
    # n_train = int(len(dataset) * train_percent)
    # n_test = len(dataset) - n_train - n_val
    # train, val, _ = random_split(dataset, [n_train, n_val, n_test])
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    # oneshot_dataset = OneshotDavisDataset('/orion/u/pancy/data/object-pursuit/davis/DAVIS', obj, resize=(256, 256))
    # oneshot_dataset = DavisDataset('/orion/u/pancy/data/object-pursuit/davis/DAVIS', obj_train, resize=(256, 256))
    # norm_dataset = DavisDataset('/orion/u/pancy/data/object-pursuit/davis/DAVIS', obj, resize=(256, 256))
    oneshot_dataset = KittiTrainDataset('/orion/u/pancy/data/object-pursuit/kitti/4')
    norm_dataset = KittiTestDataset('/orion/u/pancy/data/object-pursuit/kitti/4')
    n_train = len(oneshot_dataset)
    n_val = len(norm_dataset)
    train_loader = DataLoader(oneshot_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(norm_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
    log_writer = open(os.path.join(dir_checkpoint, "log.txt"), "w")
    global_step = 0

    info_text = f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device}
        Images scaling:  {img_scale}
        dir_img:         {dir_img}
        dir_mask:        {dir_mask}
        parameter number of the network: {sum(x.numel() for x in net.parameters() if x.requires_grad)}
    '''
    logging.info(info_text)
    log_writer.write(info_text)
    log_writer.flush()

    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-7, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    
    # if net.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    #     criterion = nn.BCEWithLogitsLoss()
        
    # Memory loss
    if args.model == 'singlenet' and use_mem_loss:    
        memloss = MemoryLoss(Base_dir='./Bases', device=device)
        mem_coeff = 0.01
        
    max_valid_acc = 0

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        count = 0
        val_list = []
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                # loss = criterion(masks_pred, true_masks)
                # loss = F.binary_cross_entropy_with_logits(masks_pred, true_masks, pos_weight=torch.tensor([get_pos_weight_from_batch(true_masks)]).to(device))
                loss = F.binary_cross_entropy_with_logits(masks_pred, true_masks)
                epoch_loss += loss.item()
                count += 1

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                if args.model == 'singlenet' and use_mem_loss:  
                    loss2 = memloss(net.hypernet, mem_coeff)
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % int(n_train / (0.05*batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                    val_score = 0
                    # val_score, _ = eval_net(net, val_loader, device)
                    acc.append(val_score)
                    val_list.append(val_score)
                    # scheduler.step(val_score)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        log_writer.write('Validation cross entropy: {}\n'.format(val_score))
                        log_writer.flush()
                    else:
                        try:
                            print("\n current coeffs: ", net.coeffs)
                        except Exception:
                            pass
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        log_writer.write('Validation Dice Coeff: {}\n'.format(val_score))
                        log_writer.flush()

        if len(val_list) > 0:
            avg_valid_acc = sum(val_list)/len(val_list)
            vis_predict(os.path.join(dir_checkpoint, 'viz_pred'), net, val_loader, device)
            if avg_valid_acc > max_valid_acc:
                vis_predict(os.path.join(dir_checkpoint, 'viz_pred'), net, val_loader, device)
                if save_cp:
                    if args.model == 'singlenet':    
                        torch.save(net.state_dict(), os.path.join(dir_checkpoint, f'Best.pth'))
                        # net.save_z(f'./Bases/{obj}.json')
                    log_writer.write(f'Checkpoint {epoch + 1} saved ! current validation accuracy: {avg_valid_acc}, current loss {epoch_loss/count}\n')
                    logging.info(f'Checkpoint {epoch + 1} saved !')
                max_valid_acc = avg_valid_acc

    log_writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0004,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--cuda', dest='cuda', type=int, default=0,
                        help='cuda device number')
    parser.add_argument('--model', type=str, default='singlenet',
                        choices=['coeffnet', 'deeplab', 'unet', 'singlenet'],
                        help='model name')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    
    if args.model == "unet":
        net = UNet(n_channels=3, n_classes=1, bilinear=True)
    elif args.model == "deeplab":
        net = DeepLab(num_classes = 1, backbone = 'resnetsub', output_stride = 16, freeze_backbone=False, pretrained_backbone=True)
    elif args.model == "singlenet":
        net = Singlenet(z_dim=100, device=device)
        path = "./checkpoints_conv_small_full/checkpoint.pth"
        net.init_hypernet(path, freeze=True)
        net.init_backbone(path)
    elif args.model == "coeffnet":
        path = "./checkpoints_conv_small_full/checkpoint.pth"
        net = Coeffnet(base_dir='./Bases', z_dim=100, device=device, hypernet_path=path)
    else:
        raise NotImplementedError
    
    logging.info(f'Network:\n'
        f'\t{net.n_channels} input channels\n'
        f'\t{net.n_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(args=args,
                  net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        # torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        print(acc)
        try:
            print("coefficients from coeffnet: ", net.coeffs)
            sys.exit(0)
        except SystemExit:
            os._exit(0)
