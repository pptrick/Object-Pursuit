import os
import json
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from dataset.basic_dataset import BasicDataset

class MultiJointDataset(Dataset):
    def __init__(self, img_dirs, mask_dirs, resize=None, random_crop=True):
        super(MultiJointDataset, self).__init__()
        self.img_dirs, self.mask_dirs = self._check_dirs(img_dirs, mask_dirs)
        self.class_num = len(self.img_dirs)
        self._init_datasets(resize=resize, random_crop=random_crop)
    
    def _check_dirs(self, img_dirs, mask_dirs):
        # check validity for img_dirs and mask_dirs
        # both img_dirs and mask_dirs should be list of same length, each dir relates to a class of Object
        assert isinstance(img_dirs, list) and isinstance(mask_dirs, list)
        assert len(img_dirs) == len(mask_dirs)
        for i in range(len(img_dirs)):
            assert os.path.isdir(img_dirs[i]) and os.path.isdir(mask_dirs[i])
        return img_dirs, mask_dirs
    
    def _init_datasets(self, resize, random_crop):
        self.datasets = []
        self.class_index_list = [] # start from zero, record class for each data sample; e.g. [0,0,0,0,1,1,2,2,2,2,3,3,3,...]
        for i in range(self.class_num):
            # TODO: set more control param
            # for each object, use a Basic Dataset as container
            dataset = BasicDataset(
                imgs_dir=self.img_dirs[i],
                masks_dir=self.mask_dirs[i],
                resize=resize,
                random_crop=random_crop
            )
            # save a dict for each object (dataset)
            self.datasets.append({
                "dataset": dataset,
                "index": i,
                "img_dir": self.img_dirs[i],
                "mask_dir": self.mask_dirs[i],
                "len": len(dataset),
                "start_index": len(self.class_index_list)
            })
            # save class index
            self.class_index_list += [i] * len(dataset)
            
    def __getitem__(self, index):
        cls = self.class_index_list[index] # dataset's index
        dataset_info = self.datasets[cls]
        index_bias = index - dataset_info["start_index"] # index in dataset
        sample = dataset_info["dataset"][index_bias]
        # add class index to sample
        sample["cls"] = cls
        return sample
    
    def __len__(self):
        return len(self.class_index_list)
    
    # public
    def getIndexList(self):
        return self.class_index_list
    
class MultiJointSampler(Sampler):
    def __init__(self, data_source, batch_size, num_balance=False):
        """This is a batch sampler for Multi-Joint Pretrain DataLoader.
        The core idea is to make each batch contains data of only one object (of the same class). However, object classes should be randomly distributed over batches.
        This sampler returns a batch of indices at a time

        Args:
            data_source (torch.utils.data.Dataset): The Multi-Joint Dataset used to construct the Dataloader
            batch_size (int): batch size
        """
        super().__init__(data_source)
        assert isinstance(data_source, MultiJointDataset)
        self.dataset_info = data_source.datasets
        self.batch_size = batch_size
        self.num_balance = num_balance
        self.length = len(self.init_index_list(self.dataset_info, self.batch_size, self.num_balance))
        
    def init_index_list(self, dataset_info, batch_size, num_balance):
        index_list = self._get_index_list(dataset_info)
        index_list = self._shuffle_each(index_list)
        index_list = self._to_batches(index_list, batch_size)
        if num_balance:
            index_list = self._num_balance(index_list)
        index_list = self._flatten(index_list)
        random.shuffle(index_list)
        return index_list
        
    def _get_index_list(self, dataset_info): # get index list from dataset
        index_list = []
        for ds in dataset_info: # for each obj
            index_list.append(list(range(ds["start_index"], ds["start_index"]+ds["len"])))
        return index_list
    
    def _shuffle_each(self, index_list):
        for i in range(len(index_list)):
            random.shuffle(index_list[i])
        return index_list
    
    def _to_batches(self, index_list, batch_size):
        new_index_list = []
        for idx_list in index_list:
            batch_num = len(idx_list) // batch_size
            if batch_num > 0:
                batches = np.array_split(idx_list[:batch_size*batch_num], batch_num)
                new_index_list.append(batches)
        return new_index_list

    def _num_balance(self, index_list):
        min_len = min([len(l) for l in index_list])
        for i in range(len(index_list)):
            index_list[i] = index_list[i][:min_len]
        return index_list
    
    def _flatten(self, index_list):
        new_index_list = []
        for i in index_list:
            for j in i:
                new_index_list.append(j)
        return new_index_list
            
    def __iter__(self):
        # init
        index_list = self.init_index_list(self.dataset_info, self.batch_size, self.num_balance)
        for idx_batch in index_list:
            yield idx_batch
            
    def __len__(self):
        return self.length


def _Davis_Multi(data_dir, trainset_only=False):
    # sample data_dir: [dir to davis]/DAVIS
    img_path = "JPEGImages"
    mask_path = "Annotations"
    res = "480p"
    val_obj = ["blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl", "dog", "drift-chicane", "drift-straight", "goat", "horsejump-high", "kite-surf", "libby", "motocross-jump", "paragliding-launch", "parkour", "scooter-black", "soapbox"]
    objects = sorted(os.listdir(os.path.join(data_dir, img_path, res)))
    if trainset_only:
        objects = [obj for obj in objects if obj not in val_obj]
    img_dirs = [os.path.join(data_dir, img_path, res, obj) for obj in objects]
    mask_dirs = [os.path.join(data_dir, mask_path, res, obj) for obj in objects]
    return img_dirs, mask_dirs

def _iThor_Multi(data_dir, trainset_only=False):
    # sample data_dir: [dir to ithor]/ithor/Pretrain/
    assert os.path.isdir(data_dir)
    objects = [obj for obj in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, obj))]
    img_dirs = [os.path.join(data_dir, obj, "imgs") for obj in objects]
    mask_dirs = [os.path.join(data_dir, obj, "masks") for obj in objects]
    return img_dirs, mask_dirs

def _VOS_Multi(data_dir, trainset_only=False):
    img_dir = os.path.join(data_dir, "JPEGImages")
    mask_dir = os.path.join(data_dir, "Annotations")
    video_idx = sorted(os.listdir(img_dir))
    out_img_dirs, out_mask_dirs = [], []
    category = {}
    for seq in video_idx:
        with open(os.path.join(mask_dir, seq, "meta.json"), 'r') as f:
            meta_info = json.load(f)["objects"]
            if len(meta_info)==1 and "1" in meta_info:
                if len(meta_info["1"]["frames"])>=32: # select sequences whose frames are more than 32
                    out_img_dirs.append(os.path.join(img_dir, seq))
                    out_mask_dirs.append(os.path.join(mask_dir, seq))
                    cat = meta_info["1"]["category"]
                    if cat in category:
                        category[cat] += 1
                    else:
                        category[cat] = 1
            else:
                continue
    print(category)
    return out_img_dirs, out_mask_dirs

def genDataLoader(dataset, data_dir, batch_size, resize=None, num_workers=1, num_balance=False, random_crop=True, trainset_only=False):
    assert len(dataset) == len(data_dir)
    dataset_map = {
        "DAVIS": _Davis_Multi,
        "iThor": _iThor_Multi,
        "VOS": _VOS_Multi,
    }
    img_dirs, mask_dirs = [], []
    i = 0
    for ds in dataset:
        if ds in dataset_map:
            _img_dirs, _mask_dirs = dataset_map[ds](data_dir[i], trainset_only)
            img_dirs += _img_dirs
            mask_dirs += _mask_dirs
            i += 1
    dataset = MultiJointDataset(img_dirs, mask_dirs, resize=resize, random_crop=random_crop)
    sampler = MultiJointSampler(dataset, batch_size, num_balance=num_balance)
    return DataLoader(dataset, num_workers=num_workers, batch_sampler=sampler), dataset
            