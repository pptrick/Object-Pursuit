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
    net.train()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred, _ = net(imgs, ident)

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
    checkpoint_path = './checkpoints_conv_small_full/checkpoint.pth'
    data_path = "/orion/u/pancy/data/object-pursuit/ithor/FloorPlan2"
    prefix = "data_FloorPlan2_"
    cuda = 0
    data_size = 400
    # obj_map = collections.OrderedDict([('SprayBottle', 0), ('Cup', 1), ('Vase', 2), ('BasketBall', 3), ('TennisRacket', 4), ('Laptop', 5), ('Ladle', 6), ('SoapBar', 7), ('WateringCan', 8), ('Bread', 9), ('Box', 10), ('Plate', 11), ('Bowl', 12), ('Book', 13), ('Kettle', 14), ('Egg', 15), ('PepperShaker', 16), ('Potato', 17), ('Pan', 18), ('TeddyBear', 19), ('Pot', 20), ('Spatula', 21), ('Plunger', 22), ('Knife', 23), ('CD', 24), ('KeyChain', 25), ('AluminumFoil', 26), ('Fork', 27), ('CellPhone', 28), ('Pen', 29), ('TissueBox', 30), ('DishSponge', 31), ('BaseballBat', 32), ('Dumbbell', 33), ('SaltShaker', 34), ('Footstool', 35), ('Mug', 36), ('Apple', 37), ('AlarmClock', 38), ('Cup2', 39), ('CreditCard', 40), ('Candle', 41), ('ScrubBrush', 42), ('Cloth', 43), ('Newspaper', 44), ('Towel', 45), ('Spoon', 46), ('SoapBottle', 47), ('ButterKnife', 48), ('Lettuce', 49), ('Tomato', 50), ('Watch', 51)]) 
    # obj_map = collections.OrderedDict([('Spatula', 0), ('CreditCard', 1), ('SaltShaker', 2), ('Pen', 3), ('SoapBottle', 4), ('Book', 5), ('Box', 6), ('TennisRacket', 7), ('Watch', 8), ('Towel', 9), ('TeddyBear', 10), ('Vase', 11), ('Bowl', 12), ('WateringCan', 13), ('SprayBottle', 14), ('AluminumFoil', 15), ('PepperShaker', 16), ('Apple', 17), ('Pan', 18), ('DishSponge', 19), ('Bread', 20), ('Ladle', 21), ('BasketBall', 22), ('Tomato', 23), ('AlarmClock', 24), ('Dumbbell', 25), ('SoapBar', 26), ('Cup', 27), ('Fork', 28), ('Cup2', 29), ('Mug', 30), ('ButterKnife', 31), ('Pot', 32), ('ScrubBrush', 33), ('Potato', 34), ('TissueBox', 35), ('CellPhone', 36), ('Spoon', 37), ('Candle', 38), ('Kettle', 39), ('KeyChain', 40), ('Cloth', 41), ('BaseballBat', 42), ('Newspaper', 43), ('Knife', 44), ('Plunger', 45), ('Lettuce', 46), ('Laptop', 47), ('CD', 48), ('Footstool', 49), ('Plate', 50), ('Egg', 51)]) 
    obj_map = collections.OrderedDict([('WateringCan', 0), ('Mug', 1), ('CD', 2), ('PepperShaker', 3), ('Knife', 4), ('ButterKnife', 5), ('TeddyBear', 6), ('Laptop', 7), ('CreditCard', 8), ('Cup2', 9), ('SoapBar', 10), ('Spatula', 11), ('Book', 12), ('Pan', 13), ('Lettuce', 14), ('Plate', 15), ('Bowl', 16), ('Box', 17), ('Vase', 18), ('Watch', 19), ('Kettle', 20), ('BaseballBat', 21), ('Candle', 22), ('DishSponge', 23), ('Pen', 24), ('SprayBottle', 25), ('Newspaper', 26), ('Cloth', 27), ('CellPhone', 28), ('Plunger', 29), ('Egg', 30), ('Cup', 31), ('BasketBall', 32), ('Dumbbell', 33), ('ScrubBrush', 34), ('Potato', 35), ('SaltShaker', 36), ('Footstool', 37), ('Apple', 38), ('Spoon', 39), ('SoapBottle', 40), ('KeyChain', 41), ('AluminumFoil', 42), ('Pot', 43), ('Bread', 44), ('Fork', 45), ('Ladle', 46)])
    # obj_map = collections.OrderedDict([('Spatula', 0), ('Vase', 1), ('PepperShaker', 2), ('Pan', 3), ('AlarmClock', 4), ('Cup', 5), ('ButterKnife', 6), ('Kettle', 7), ('KeyChain', 8), ('Newspaper', 9), ('Laptop', 10), ('CD', 11), ('Plate', 12)])
    # obj_map = collections.OrderedDict([('SprayBottle', 0), ('Cup', 1), ('Vase', 2), ('BasketBall', 3), ('TennisRacket', 4), ('Laptop', 5), ('Ladle', 6), ('SoapBar', 7), ('WateringCan', 8), ('Bread', 9), ('Box', 10), ('Plate', 11), ('Bowl', 12), ('Book', 13), ('Kettle', 14), ('Egg', 15), ('PepperShaker', 16), ('Potato', 17), ('Pan', 18), ('TeddyBear', 19), ('Pot', 20), ('Spatula', 21), ('Plunger', 22), ('Knife', 23), ('CD', 24), ('KeyChain', 25), ('AluminumFoil', 26), ('Fork', 27), ('CellPhone', 28), ('Pen', 29), ('TissueBox', 30), ('DishSponge', 31), ('BaseballBat', 32), ('Dumbbell', 33), ('SaltShaker', 34), ('Footstool', 35), ('Mug', 36), ('Apple', 37), ('AlarmClock', 38), ('Cup2', 39), ('CreditCard', 40), ('Candle', 41), ('ScrubBrush', 42), ('Cloth', 43), ('Newspaper', 44), ('Towel', 45), ('Spoon', 46), ('SoapBottle', 47), ('ButterKnife', 48), ('Lettuce', 49), ('Tomato', 50), ('Watch', 51)]) 
    
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
    