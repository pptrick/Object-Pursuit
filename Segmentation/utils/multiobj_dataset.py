import os
import collections
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import utils.custom_transforms as tr 
from torch.utils.data.sampler import Sampler

class MultiobjDataset(Dataset):
    def __init__(self, data_dir, resize = None, prefix="data_FloorPlan2_"):
        super(MultiobjDataset, self).__init__()
        self.data_dir = data_dir
        self.prefix = prefix
        self.resize = resize
        self._parse_class(data_dir, prefix)
        self._parse_file()
        
    def _parse_class(self, data_dir, prefix):
        self.class_dirs = [d for d in os.listdir(data_dir) if d.startswith(prefix)]
        self.classes = collections.OrderedDict()
        counter = 0
        for d in self.class_dirs:
            cls = d[len(prefix):]
            if cls not in self.classes:
                self.classes[cls] = counter
                counter += 1
        self.class_dirs = [os.path.join(data_dir, d) for d in self.class_dirs]
        self.obj_num = len(self.classes)
        print("object ident mapping: ", self.classes)
        
    def _parse_file(self):
        self.img_files = []
        self.mask_files = []
        self.cls_list = []
        self.index_list = []
        ptr = 0
        for cls_dir in self.class_dirs:
            img_dir = os.path.join(cls_dir, 'imgs')
            mask_dir = os.path.join(cls_dir, 'masks')
            cls = os.path.basename(cls_dir)[len(self.prefix):]
            self.img_files += [os.path.join(img_dir, f) for f in os.listdir(img_dir) if (f.endswith(".png") or f.endswith(".jpg"))]
            self.mask_files += [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if (f.endswith(".png") or f.endswith(".jpg"))]
            self.cls_list += [self.classes[cls]]*len(os.listdir(img_dir))
            assert len(self.img_files) == len(self.mask_files) and len(self.cls_list) == len(self.mask_files)
            self.index_list.append(list(range(ptr, len(self.cls_list))))
            ptr = len(self.cls_list)
            
    def _make_img_gt_point_pair(self, index):
        mask_file = self.mask_files[index]
        img_file = self.img_files[index]
        
        assert os.path.basename(img_file) == os.path.basename(mask_file)
        
        _img = Image.open(img_file).convert('RGB')
        _mask = Image.open(mask_file)
        
        if self.resize is not None:
            _img = _img.resize(self.resize)
            _mask = _mask.resize(self.resize)
        
        assert _img.size == _mask.size, \
            f'Image and mask {img_file} should be the same size, but are {_img.size} and {_mask.size}'

        return _img, _mask
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.MaskExpand(),
            tr.ImgNorm(),
            tr.ToTensor()])
        return composed_transforms(sample)

            
    def __getitem__(self, index):
        img, mask = self._make_img_gt_point_pair(index)
        sample = {'image': img, 'mask': mask}
        sample = self.transform_tr(sample)
        sample['cls'] = self.cls_list[index]
        return sample
    

class SameClassSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super(SameClassSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.index_list = data_source.index_list
        assert isinstance(self.index_list, list)
        self._len, self._min_len = self._cal_len()
        
    def _cal_len(self):
        l = 0
        min_len = min([len(ls) for ls in self.index_list])
        l = (min_len//self.batch_size) * len(self.index_list)
        return l, min_len
        
    def _unstop_list(self, cls_ptr):
        unstop_list = []
        for i in range(len(cls_ptr)):
            if cls_ptr[i] <= (self._min_len - self.batch_size):
                unstop_list.append(i)
        return unstop_list
        
    def __iter__(self):
        # shuffle
        for i in range(len(self.index_list)):
            random.shuffle(self.index_list[i])
        cls_ptr = [0] * len(self.index_list)
        unstop_list = self._unstop_list(cls_ptr)
        ptr = 0
        while len(unstop_list) > 0:
            if ptr % len(cls_ptr) in unstop_list:
                i = ptr % len(cls_ptr)
            else:
                i = random.choice(unstop_list)
            batch = self.index_list[i][cls_ptr[i]:cls_ptr[i]+self.batch_size]
            assert len(batch) == self.batch_size
            yield batch
            cls_ptr[i] += self.batch_size
            unstop_list = self._unstop_list(cls_ptr)
            ptr += 1
            
    def __len__(self):
        return self._len
        
        
def Multiobj_Dataloader(data_dir, batch_size, num_workers=1, prefix="data_FloorPlan3_", resize = None):
    ds = MultiobjDataset(data_dir=data_dir, prefix=prefix, resize=resize)
    sampler = SameClassSampler(ds, batch_size)
    return DataLoader(ds, num_workers=num_workers, batch_sampler=sampler), ds
        
    
if __name__ == "__main__":
    dataloader, _ = Multiobj_Dataloader("/data/pancy/iThor/single_obj/FloorPlan3", batch_size=4, prefix="data_FloorPlan3_")
    print(len(dataloader))
    for step, data in enumerate(dataloader):
        print(data['cls'])
            
            
        
        
        