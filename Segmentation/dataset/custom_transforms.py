import torch
import random
import numpy as np

from PIL import Image

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'mask': mask}
        
class ImgNorm(object):
    """Normalize image from [0, 255] to [0, 1]"""
    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        if img.max() > 1:
            img /= 255.0

        return {'image': img,
                'mask': mask}
        
class MaskExpand(object):
    """For some bool mask, expand it on dim num_classes"""
    
    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        #expand
        if len(mask.shape) == 3:
            print("warning, find a 3-dim mask")
            mask = mask[:,:,0]
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        mask = mask.transpose((2,0,1))
        if mask.max() > 1:
            mask = mask/255
        
        return {'image': img,
                'mask': mask}
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['mask']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'mask': mask}