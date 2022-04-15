import os
import os.path as osp
import numpy as np
import torch
import random
from dataset.color_jitter import ColorJitter
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import dataset.custom_transforms as tr 

crop_box = [430, 0, 805, 375]
# crop_box = [600, 0, 1242, 375]

class KittiTrainDataset(Dataset):
    def __init__(self, kitti_dir, dataset_len=32, resize=None):
        self.dataset_len = dataset_len
        self.resize = resize
        super(KittiTrainDataset, self).__init__()
        self.train_dir = osp.join(kitti_dir, 'train')
        assert osp.isdir(self.train_dir)
        self.image = osp.join(self.train_dir, 'rgb.png')
        self.instance_mask = osp.join(self.train_dir, 'instance.png')
        assert osp.isfile(self.image) and osp.isfile(self.instance_mask)
        self.cj = ColorJitter(brightness=0.1, contrast=0.1, sharpness=0.1, color=0.1)
        
    def _augment(self, img, mask):
        img_size = img.size
        # color jitter
        img = self.cj(img)
        # img flip
        if random.random()<0.4:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        # random crop
        crop_rate = 0.3
        delta_W, delta_H = int(crop_rate*img_size[0]), int(crop_rate*img_size[1])
        delta_w, delta_h = random.randint(0, delta_W), random.randint(0, delta_H)
        delta_x, delta_y = random.randint(0, delta_w), random.randint(0, delta_h)
        img = img.crop([delta_x, delta_y, delta_x+img_size[0]-delta_w, delta_y+img_size[1]-delta_h]).resize(img_size)
        mask = mask.crop([delta_x, delta_y, delta_x+img_size[0]-delta_w, delta_y+img_size[1]-delta_h]).resize(img_size)
        return img, mask
        
    def _make_img_gt_point_pair(self, index, crop=True):
        _img = Image.open(self.image).convert('RGB')
        _mask = Image.open(self.instance_mask)
        
        assert _img.size == _mask.size, \
            f'Image and mask {index} should be the same size, but are {_img.size} and {_mask.size}'
            
        if crop:
            _img = _img.crop(crop_box)
            _mask = _mask.crop(crop_box)
            
        # _img, _mask = self._augment(_img, _mask)
        
        if self.resize is not None:
            _img = _img.resize(self.resize)
            _mask = _mask.resize(self.resize)
        
        _img = np.array(_img).astype(np.float32)
        _mask = np.array(_mask).astype(np.float32)
        
        # _mask = _mask == _mask[210][360]
        _mask = _mask == _mask[201][198]
        
        return _img, _mask, self.image, self.instance_mask
    
    def __getitem__(self, index):
        img, mask, img_file, mask_file = self._make_img_gt_point_pair(index)
        sample = {'image': img, 'mask': mask}
        sample = self.transform_tr(sample)
        
        mask = sample['mask']
        
        sample['img_file'] = img_file
        sample['mask_file'] = mask_file
        return sample
    
    def __len__(self):
        return self.dataset_len
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.MaskExpand(),
            tr.ImgNorm(),
            tr.ToTensor()])
        return composed_transforms(sample)
    
class KittiTestDataset(Dataset):
    def __init__(self, kitti_dir, resize=None):
        super(KittiTestDataset, self).__init__()
        self.resize = resize
        self.test_dir = osp.join(kitti_dir, "data")
        assert osp.isdir(self.test_dir)
        self.test_files = [osp.join(self.test_dir, img_f) for img_f in sorted(os.listdir(self.test_dir)) if img_f.endswith('.png')]
    
    def _make_img_gt_point_pair(self, index, crop=True):
        _img = Image.open(self.test_files[index]).convert('RGB')
        
        if crop:
            _img = _img.crop(crop_box)
        
        if self.resize is not None:
            _img = _img.resize(self.resize)
        
        _img = np.array(_img).astype(np.float32)
        if _img.max() > 1:
            _img /= 255.0
        _img = np.array(_img).astype(np.float32).transpose((2, 0, 1))
        _img = torch.from_numpy(_img).float()
        
        return _img, self.test_files[index]
    
    def __getitem__(self, index):
        img, img_file = self._make_img_gt_point_pair(index)
        sample = {'image': img, 'img_file': img_file}
        return sample
    
    def __len__(self):
        return len(self.test_files)
    
if __name__ == "__main__":
    kds = KittiTestDataset("/orion/u/pancy/data/object-pursuit/kitti/1")
    batch = kds[1]
    img = batch['image']
    f = batch['img_file']
    print(f, img.size())