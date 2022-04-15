import os
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def mask_on_img(img, mask, alpha=1.0):
    """overlap mask on img

    Args:
        img (PIL Image): input target rgb image, H*W*C
        mask (bool array): predict mask of img, H*W
        alpha (float, optional): transparent value of the mask. Defaults to 0.9.
    """
    res = np.array(img.copy())
    res[:, :, 0] = mask[:,:]*alpha*255 + (1-mask[:,:]*alpha)*res[:,:,0]
    return res

def vis_predict(output_dir, net, loader, device, out_threshold=0.5):
    net.eval()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    n_val = len(loader)
    counter = 0
    to_img = transforms.ToPILImage()
    with tqdm(total=n_val, desc='Visualization round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            img_files = batch['img_file']
            imgs = imgs.to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                mask_pred = net(imgs)
            
            batch_size = imgs.size(0)
            for i in range(batch_size):
                probs = torch.sigmoid(mask_pred[i])
                probs = probs.squeeze(0)
            
                full_mask = probs.squeeze().cpu().numpy()
                full_mask = full_mask > out_threshold
                
                img = to_img(imgs[i])    
                res = mask_on_img(img, full_mask)
                res_img = Image.fromarray(res)
                res_img.save(os.path.join(output_dir, os.path.basename(img_files[i])))
                counter += 1
                      
            pbar.update()
                
            
            