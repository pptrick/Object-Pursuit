import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.color_jitter import ColorJitter
from torchvision import transforms
import utils.custom_transforms as tr 


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', train=False):
        self.imgs_dir = self._parse_dirs(imgs_dir)
        self.masks_dir = self._parse_dirs(masks_dir)
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.train = train
        if train:
            self.color_jitter = ColorJitter(brightness=0.3026, contrast=0.2935, sharpness=0.736, color=0.3892)
        else:
            self.color_jitter = ColorJitter()
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        self._get_ids()
        # logging.info(f'Creating dataset with {len(self.ids)} examples')
        print(f'Creating dataset with {len(self.ids)} examples')
        
    def _parse_dirs(self, dirs):
        if type(dirs) == list:
            return dirs
        elif type(dirs) == str:
            return [dirs]
        else:
            return None
        
    def _get_ids(self):
        self.ids = [] # each data specified with a file path
        for img_dir in self.imgs_dir:
            if img_dir.endswith("/"):
                root = os.path.dirname(os.path.dirname(img_dir))
            else:
                root = os.path.dirname(img_dir)
            mask_dir = os.path.join(root, "masks")
            if mask_dir in self.masks_dir or (mask_dir+'/') in self.masks_dir:
                idx = [splitext(file)[0] for file in listdir(img_dir) if not file.startswith('.')]
                img_dirs = [img_dir] * len(idx)
                mask_dirs = [mask_dir] * len(idx)
                self.ids += zip(idx, img_dirs, mask_dirs)
            else:
                print("[Warning] can find mask dir: ", mask_dir)

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
    
    def _make_img_gt_point_pair(self, index):
        idx = self.ids[index]
        mask_file = glob(os.path.join(os.path.join(idx[2], idx[0] + self.mask_suffix + '.*')))
        img_file = glob(os.path.join(idx[1], idx[0] + '.*'))
        
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        _img = Image.open(img_file[0]).convert('RGB')
        _mask = Image.open(mask_file[0])
        
        assert _img.size == _mask.size, \
            f'Image and mask {idx} should be the same size, but are {_img.size} and {_mask.size}'

        return _img, _mask, img_file[0], mask_file[0]

    def __getitem__(self, i):
        img, mask, img_file, mask_file = self._make_img_gt_point_pair(i)
        sample = {'image': img, 'mask': mask}
        
        if self.train:
            sample = self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)
            
        sample['img_file'] = img_file
        sample['mask_file'] = mask_file
        return sample
        
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.MaskExpand(),
            tr.ImgNorm(),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)
    
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.MaskExpand(),
            tr.ImgNorm(),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)



        
if __name__ == "__main__":
    d = BasicDataset(["/data/pancy/iThor/single_obj/data_FloorPlan4_Egg/imgs/", "/data/pancy/iThor/single_obj/data_FloorPlan3_Egg/imgs/"], ["/data/pancy/iThor/single_obj/data_FloorPlan3_Egg/masks/", "/data/pancy/iThor/single_obj/data_FloorPlan4_Egg/masks/"])
    print(d[3])
