import os
import numpy as np
import random
import json
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.color_jitter import ColorJitter
import dataset.custom_transforms as tr 

class VosDataset(Dataset):
    def __init__(self, dataset_dir, obj, resize=None, mode='instancewise',
                 cat2video='/orion/u/yzcong/datasets/train/cat2video.json',
                 repeat=None):
        super(VosDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.resize = resize
        with open(cat2video, 'r') as f:
            self.cat2video = json.load(f)

        # catwise: load all images of the same category;
        # instancewise: load all images of the same instance belonging to the category
        self.mode = mode  

        self.imgs_dirs, self.masks_dirs, self.instance_ids, self.objects = self._check_dir(dataset_dir, obj)
        self.id_list = self._get_ids(self.imgs_dirs, self.masks_dirs, self.instance_ids)
        #augmentation
        self.cj = ColorJitter(brightness=0.1, contrast=0.1, sharpness=0.1, color=0.1)
        self.repeat = repeat if repeat is not None else (500 + len(self.id_list)) // len(self.id_list)
        
    @classmethod
    def get_obj_list(self, ds_dir):
        assert os.path.isdir(ds_dir)
        sub_dirs = os.listdir(ds_dir)
        assert 'JPEGImages' in sub_dirs and 'Annotations' in sub_dirs
        img_dir = os.path.join(ds_dir, 'JPEGImages')
        mask_dir = os.path.join(ds_dir, 'Annotations')
        # res = '480p'
        # img_dir = os.path.join(img_dir, res)
        # mask_dir = os.path.join(mask_dir, res)
        obj_in_img = os.listdir(img_dir)
        obj_in_mask = os.listdir(mask_dir)
        assert len(obj_in_img) == len(obj_in_mask)
        return obj_in_img, obj_in_mask, img_dir, mask_dir
        
    def _check_dir(self, ds_dir, obj):
        obj_in_img, obj_in_mask, img_dir, mask_dir = VosDataset.get_obj_list(ds_dir)
        if self.mode == 'instancewise':
            while True:
                video_info = self.cat2video[obj][np.random.randint(0, len(self.cat2video[obj]))]
                video_id, instance_ids = video_info['video'], [video_info['id']]
                # print(instance_ids)
                if len(instance_ids[0]) > 1 or video_info['frames'] < 20:
                    continue
                else:
                    break
            print(video_info['frames'])
            assert (video_id in obj_in_img) and (video_id in obj_in_mask)
            img_dirs = [os.path.join(img_dir, video_id)]
            mask_dirs = [os.path.join(mask_dir, video_id)]
            assert len(os.listdir(img_dir)) == len(os.listdir(mask_dir))
        elif self.mode == 'catwise':
            instance_ids = []
            img_dirs = []
            mask_dirs = []
            for video_info in self.cat2video[obj]:
                video_id, instance_id = video_info['video'], video_info['id']
                if len(instance_id) > 1:
                    continue
                assert (video_id in obj_in_img) and (video_id in obj_in_mask)
                img_dirs.append(os.path.join(img_dir, video_id))
                mask_dirs.append(os.path.join(mask_dir, video_id))
                instance_ids.append(instance_id)
                assert len(os.listdir(img_dir)) == len(os.listdir(mask_dir))
        return img_dirs, mask_dirs, instance_ids, obj_in_img
    
    def _get_ids(self, imgs_dirs, masks_dirs, instance_ids):
        id_list = []
        for (imgs_dir, masks_dir), instance_id in zip(zip(imgs_dirs, masks_dirs), instance_ids):
            ids = sorted(os.listdir(imgs_dir))
            for seq_id in ids:
                img_file = os.path.join(imgs_dir, seq_id)
                mask_file = os.path.join(masks_dir, seq_id.replace('.jpg', '.png')) # file format of mask is .png
                assert os.path.isfile(img_file) and os.path.isfile(mask_file)
                id_list.append((img_file, mask_file, instance_id))
        # if (len(id_list) > 2000):
            # id_list = np.random.choice(id_list, 2000, replace=False).item()
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
    
    def _augment(self, img, mask):
        img_size = img.size
        # color jitter
        # img = self.cj(img)
        # img flip
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        # random crop
        crop_rate = 0.75
        delta_W, delta_H = int(crop_rate*img_size[0]), int(crop_rate*img_size[1])
        delta_x, delta_y = random.randint(0, img_size[0] - delta_W), random.randint(0, img_size[1] - delta_H)
        # crop_rate = 0.3
        # delta_W, delta_H = int(crop_rate*img_size[0]), int(crop_rate*img_size[1])
        # delta_w, delta_h = random.randint(0, delta_W), random.randint(0, delta_H)
        # delta_x, delta_y = random.randint(0, delta_w), random.randint(0, delta_h)
        img = img.crop([delta_x, delta_y, delta_x+delta_W, delta_y+delta_H]).resize(self.resize)
        mask = mask.crop([delta_x, delta_y, delta_x+delta_W, delta_y+delta_H]).resize(self.resize)
        return img, mask
    
    def _make_img_gt_point_pair(self, index, augment=True):
        img_file, mask_file, instance_id = self.id_list[index]
        assert len(instance_id) == 1
        instance_id = instance_id[0]

        _img = Image.open(img_file).convert('RGB')
        _mask = Image.open(mask_file)

        # _img.save("testori.png")
        # _mask.convert("P").save("testori_mask.png")
        
        assert _img.size == _mask.size, \
            f'Image and mask {index} should be the same size, but are {_img.size} and {_mask.size}'
            
        if augment:
            _img, _mask = self._augment(_img, _mask)
            
        if self.resize is not None:
            _img = _img.resize(self.resize)
            _mask = _mask.resize(self.resize)

        # print(_mask)
        _mask = np.array(_mask).astype(np.float32)
        # print(np.unique(_mask))
        instance_id = float(instance_id)
        _mask[_mask != instance_id] = 0.0
        _mask[_mask == instance_id] = 255.0
        # print(np.unique(_mask))
        # _img.save("test.png")
        # Image.fromarray(_mask).convert("L").save("test_mask.png")
            
        return _img, _mask, img_file, mask_file
        
    def __getitem__(self, index):
        index = index % len(self.id_list)
        img, mask, img_file, mask_file = self._make_img_gt_point_pair(index)
        sample = {'image': img, 'mask': mask}
        sample = self.transform_tr(sample)
        
        sample['img_file'] = img_file
        sample['mask_file'] = mask_file
        return sample
    
    def __len__(self):
        return len(self.id_list) * self.repeat
        
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.MaskExpand(),
            tr.ImgNorm(),
            tr.ToTensor()])
        return composed_transforms(sample)
    
    
class OneshotVosDataset(VosDataset):
    def __init__(self, dataset_dir, obj, index=0, dataset_len=32, resize=None):
        super().__init__(dataset_dir, obj, resize=resize)
        assert index < len(self.id_list)
        self.target_img_file, self.target_mask_file = self.id_list[index]
        self.dataset_len = dataset_len
        
    def _make_img_gt_point_pair(self, index, random_crop=False):
        _img = Image.open(self.target_img_file).convert('RGB')
        _mask = Image.open(self.target_mask_file)
        
        assert _img.size == _mask.size, \
            f'Image and mask {index} should be the same size, but are {_img.size} and {_mask.size}'
            
        if random_crop:
            _img, _mask = self._random_crop(_img, _mask)
            
        # _img, _mask = self._augment(_img, _mask)
            
        if self.resize is not None:
            _img = _img.resize(self.resize)
            _mask = _mask.resize(self.resize)
        
        return _img, _mask, self.target_img_file, self.target_mask_file
    
    def __len__(self):
        return self.dataset_len
        
    
if __name__ == "__main__":
    ds = VosDataset(dataset_dir='/orion/u/yzcong/datasets/train', obj='elephant', resize=(320, 180), mode='instancewise')
    print(len(ds))
    ds[0]