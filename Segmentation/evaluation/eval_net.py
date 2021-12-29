import torch
import torch.nn.functional as F
from tqdm import tqdm

from loss.dice_loss import dice_coeff
from loss.criterion import jaccard

def eval_net(net, loader, device, use_IOU=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.train()
    # net.eval()
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
                    if use_IOU:
                        res = jaccard(true_masks, pred)
                    else:
                        res = dice_coeff(pred, true_masks).item()
                    tot += res
                    records.append((res, img_file[0], mask_file[0]))
            pbar.update()

    net.train()
    decay = records[-1][0] - records[0][0]
    return tot / n_val, (decay, records)