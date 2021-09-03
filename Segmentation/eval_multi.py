import os
import collections
import torch
import random
import torch.nn as nn
from tqdm import tqdm

from model.coeffnet.coeffnet import Multinet
from loss.dice_loss import dice_coeff
from dataset.basic_dataset import BasicDataset
from torch.utils.data import DataLoader, sampler

import torch.nn.functional as F

def eval_net(net, loader, ident, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs, ident)

            if net.n_classes > 1:
                res = F.cross_entropy(mask_pred, true_masks).item()
                tot += res
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                res = dice_coeff(pred, true_masks).item()
                tot += res
            pbar.update()

    net.train()
    
    return tot / n_val

if __name__ == "__main__":
    checkpoint_path = './checkpoints_conv_small/checkpoint.pth'
    data_path = "/data/pancy/iThor/single_obj/small_FloorPlan2"
    prefix = "data_FloorPlan2_"
    cuda = 0
    data_size = 400
    # obj_map = collections.OrderedDict([('Spatula', 0), ('CreditCard', 1), ('SaltShaker', 2), ('Pen', 3), ('SoapBottle', 4), ('Book', 5), ('Box', 6), ('TennisRacket', 7), ('Watch', 8), ('Towel', 9), ('TeddyBear', 10), ('Vase', 11), ('Bowl', 12), ('WateringCan', 13), ('SprayBottle', 14), ('AluminumFoil', 15), ('PepperShaker', 16), ('Apple', 17), ('Pan', 18), ('DishSponge', 19), ('Bread', 20), ('Ladle', 21), ('BasketBall', 22), ('Tomato', 23), ('AlarmClock', 24), ('Dumbbell', 25), ('SoapBar', 26), ('Cup', 27), ('Fork', 28), ('Cup2', 29), ('Mug', 30), ('ButterKnife', 31), ('Pot', 32), ('ScrubBrush', 33), ('Potato', 34), ('TissueBox', 35), ('CellPhone', 36), ('Spoon', 37), ('Candle', 38), ('Kettle', 39), ('KeyChain', 40), ('Cloth', 41), ('BaseballBat', 42), ('Newspaper', 43), ('Knife', 44), ('Plunger', 45), ('Lettuce', 46), ('Laptop', 47), ('CD', 48), ('Footstool', 49), ('Plate', 50), ('Egg', 51)]) 
    # obj_map = collections.OrderedDict([('Spatula', 0)])
    obj_map = collections.OrderedDict([('Spatula', 0), ('Vase', 1), ('PepperShaker', 2), ('Pan', 3), ('AlarmClock', 4), ('Cup', 5), ('ButterKnife', 6), ('Kettle', 7), ('KeyChain', 8), ('Newspaper', 9), ('Laptop', 10), ('CD', 11), ('Plate', 12)])
    
    print(obj_map)

    device = torch.device('cuda:'+str(cuda) if torch.cuda.is_available() else 'cpu')
    net = Multinet(obj_num=len(obj_map), z_dim=100)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.to(device=device)

    print(f"load checkpoints from {checkpoint_path}")

    for obj in obj_map:
        ident = torch.tensor([obj_map[obj]]).to(device)
        data_dir = os.path.join(data_path, prefix+obj)
        img_dir = os.path.join(data_dir, "imgs")
        mask_dir = os.path.join(data_dir, "masks")
        eval_dataset = BasicDataset(img_dir, mask_dir, (256, 256))

        n_size = len(eval_dataset)
        indices = [i for i in range(n_size)]
        random.shuffle(indices)
        if data_size is not None:
            eval_sampler = sampler.SubsetRandomSampler(indices[:min(n_size, data_size)])
        else:
            eval_sampler = sampler.SubsetRandomSampler(indices[:n_size])
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=True, sampler=eval_sampler)
        
        res = eval_net(net, eval_loader, ident, device)
        print(f"Object: {obj}, identity: {ident[0].item()}, dataset size: {n_size}, avg eval coeff: {res}")
    
    # torch.save(net.state_dict(),'./checkpoints_equal/checkpoint_test.pth')
    