import os
import sys
import random
from shutil import copyfile

from utils.dataset import BasicDataset, DepthDataset
from torch.utils.data import DataLoader, sampler
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from loss.dice_loss import dice_coeff

from model.deeplabv3.deeplab import *
from model.unet import UNet
from model.coeffnet.coeffnet_deeplab import Coeffnet_Deeplab


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    records = []
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            img_file, mask_file = batch['img_file'], batch['mask_file']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                res = F.cross_entropy(mask_pred, true_masks).item()
                tot += res
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                res = dice_coeff(pred, true_masks).item()
                tot += res
                records.append((res, img_file[0], mask_file[0]))
            pbar.update()

    net.train()
    
    return tot / n_val, records

def _get_args():
    parser = argparse.ArgumentParser(description='Evaluate the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dir', dest='dir', type=str,
                        help="Load img and mask from dir")
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-n', '--size', dest='data_size', type=int, default=None,
                        help='Size of testing data')
    parser.add_argument('-r', '--record', dest='record', type=str, default=None,
                        help='dir to save record (top performance and worse performance)')

    return parser.parse_args()

def parse_load(load):
    if os.path.isfile(load):
        return [load]
    elif os.path.isdir(load):
        return [os.path.join(load, file) for file in sorted(os.listdir(load)) if file.endswith(".pth")]
    else:
        return []
    
def save_record(records, dir, rate=0.01):
    if not os.path.exists(dir):
        os.mkdir(dir)
    records.sort(key = lambda x: float(x[0]))
    num = rate * len(records)
    # poor files
    poor_dir = os.path.join(dir, "poor")
    if not os.path.exists(poor_dir):
        os.mkdir(poor_dir)
    for rec in records[:int(num)]:
        src_file = rec[1]
        file_rt, suffix = os.path.splitext(os.path.basename(src_file))
        target_file = os.path.join(poor_dir, file_rt+"_"+str(rec[0])+suffix)
        copyfile(src_file, target_file)
    # good files
    good_dir = os.path.join(dir, "good")
    if not os.path.exists(good_dir):
        os.mkdir(good_dir)
    for rec in records[len(records)-int(num):]:
        src_file = rec[1]
        file_rt, suffix = os.path.splitext(os.path.basename(src_file))
        target_file = os.path.join(good_dir, file_rt+"_"+str(rec[0])+suffix)
        copyfile(src_file, target_file)
    
    

if __name__ == '__main__':
    args = _get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # model
    # net = UNet(n_channels=3, n_classes=1, bilinear=True)
    # net = DeepLab(num_classes = 1, backbone = 'resnetsub', output_stride = 16, freeze_backbone=True)
    net = Coeffnet_Deeplab("/home/pancy/IP/Object-Pursuit/Segmentation/Bases", device)
    
    print(f'Network:\n'
        f'\t{net.n_channels} input channels\n'
        f'\t{net.n_classes} output channels (classes)\n')
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

    
    # data
    img_dir = os.path.join(args.dir, "imgs")
    mask_dir = os.path.join(args.dir, "masks")
    eval_dataset = BasicDataset(img_dir, mask_dir, args.scale)

    n_size = len(eval_dataset)
    indices = [i for i in range(n_size)]
    random.shuffle(indices)
    if args.data_size is not None:
        eval_sampler = sampler.SubsetRandomSampler(indices[:min(n_size, args.data_size)])
    else:
        eval_sampler = sampler.SubsetRandomSampler(indices[:n_size])
    eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True, drop_last=True, sampler=eval_sampler)
    print("[Info] testing data size: ", len(eval_loader))

    try:
        if not args.load:
            net.to(device=device)
            res, records = eval_net(net, eval_loader, device)
            print("average dice coeff: ", res)
        else:
            results = []
            model_checkpoints = parse_load(args.load)
            for cp in model_checkpoints:
                net.load_state_dict(
                    torch.load(cp, map_location=device)
                )
                print(f'Model loaded from {cp}')
                net.to(device=device)
                res, records = eval_net(net, eval_loader, device)
                if args.record is not None:
                    save_record(records, args.record)
                print("average dice coeff: ", res)
                results.append(res)
            if len(results) > 0:
                print("==================================")
                print("max test dice coeff: ", max(results))
                print("all results: ", results)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

