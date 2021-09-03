import os
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import dataset.custom_transforms as tr 

class DavisDataset(Dataset):
    def __init__(self, dataset_dir, obj, resize=None):
        super(DavisDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.resize = resize
        self.imgs_dir, self.masks_dir, self.objects = self._check_dir(dataset_dir, obj)
        self.id_list = self._get_ids(self.imgs_dir, self.masks_dir)
        
        
    def _check_dir(self, ds_dir, obj):
        assert os.path.isdir(ds_dir)
        sub_dirs = os.listdir(ds_dir)
        assert 'JPEGImages' in sub_dirs and 'Annotations' in sub_dirs
        img_dir = os.path.join(ds_dir, 'JPEGImages')
        mask_dir = os.path.join(ds_dir, 'Annotations')
        res = os.listdir(img_dir)[0]
        img_dir = os.path.join(img_dir, res)
        mask_dir = os.path.join(mask_dir, res)
        obj_in_img = os.listdir(img_dir)
        obj_in_mask = os.listdir(mask_dir)
        assert (obj in obj_in_img) and (obj in obj_in_mask)
        img_dir = os.path.join(img_dir, obj)
        mask_dir = os.path.join(mask_dir, obj)
        assert len(os.listdir(img_dir)) == len(os.listdir(mask_dir))
        return img_dir, mask_dir, obj_in_img
    
    def _get_ids(self, imgs_dir, masks_dir):
        id_list = []
        ids = sorted(os.listdir(imgs_dir))
        for id in ids:
            img_file = os.path.join(imgs_dir, id)
            mask_file = os.path.join(masks_dir, id.replace('.jpg', '.png')) # file format of mask is .png
            assert os.path.isfile(img_file) and os.path.isfile(mask_file)
            id_list.append((img_file, mask_file))
        return id_list
    
    def _random_crop(self, img, mask):
        length = min(img.size[0], img.size[1])
        bias = random.randint(0, max(img.size[0], img.size[1])-length)
        if img.size[0] > length:
            img = img.crop([bias, 0, bias+length, length])
            mask = mask.crop([bias, 0, bias+length, length])
        else:
            img = img.crop([0, bias, length, bias+length])
            mask = mask.crop([0, bias, length, bias+length])
        return img, mask
    
    def _make_img_gt_point_pair(self, index, random_crop=True):
        img_file, mask_file = self.id_list[index]
        _img = Image.open(img_file).convert('RGB')
        _mask = Image.open(mask_file)
        
        assert _img.size == _mask.size, \
            f'Image and mask {index} should be the same size, but are {_img.size} and {_mask.size}'
            
        if random_crop:
            _img, _mask = self._random_crop(_img, _mask)
            
        if self.resize is not None:
            _img = _img.resize(self.resize)
            _mask = _mask.resize(self.resize)
            
        img = np.array(_img).astype(np.float32)
        mask = np.array(_mask).astype(np.float32)
        mask = mask == 1
        
        return img, mask
        
    def __getitem__(self, index):
        img, mask = self._make_img_gt_point_pair(index)
        sample = {'image': img, 'mask': mask}
        sample = self.transform_tr(sample)
        return sample
        
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.MaskExpand(),
            tr.ImgNorm(),
            tr.ToTensor()])
        return composed_transforms(sample)
        
    
if __name__ == "__main__":
    ds = DavisDataset(dataset_dir='/data/pancy/Davis/DAVIS-2017-trainval-480p/DAVIS', obj='bmx-bumps', resize=(256, 256))
    ds[0]