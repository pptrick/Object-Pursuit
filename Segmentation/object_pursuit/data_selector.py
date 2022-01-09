import os
import copy
import random

from dataset.basic_dataset import BasicDataset
from dataset.davis_dataset import DavisDataset

class iThorDataSelector(object):    
    def __init__(self, data_dir, strat="sequence", resize=None, shuffle_seed=None, insert_seen=True, limit_num=None):
        assert os.path.isdir(data_dir)
        self.strat = strat
        self.resize = resize
        self.data_dir = data_dir
        self.dir_path = self._get_obj_paths(shuffle_seed, insert_seen, limit_num)
        self.counter = 0
        
    def _get_obj_paths(self, shuffle_seed=None, insert_seen=True, limit_num=None):
        dir_names = sorted(os.listdir(self.data_dir))
        dir_names = self._shuffle(dir_names, shuffle_seed)
        if insert_seen:
            dir_names = self._insert_seen_object(dir_names)
        print("[iThor Data Selector] object list: ", dir_names)
        dir_path = [os.path.join(self.data_dir, dn) for dn in dir_names]
        if type(limit_num) == int and limit_num > 0:
            dir_path = dir_path[0:limit_num]
        return dir_path
        
    def _shuffle(self, object_list, shuffle_seed=None):
        if shuffle_seed is not None and self.strat == "sequence":
            r = random.random
            random.seed(shuffle_seed)
            random.shuffle(object_list, random=r)
        elif self.strat == "random":
            random.shuffle(object_list)
        return object_list
    
    def _insert_seen_object(self, object_list, interval=7):
        if len(object_list) <= 0:
            return []
        new_object_list = []
        count = 0
        rec = object_list[0]
        for obj in object_list:
            count += 1
            new_object_list.append(obj)
            if count % interval == 0:
                new_object_list.append(rec)
                rec = object_list[count]
        return new_object_list
    
    def _get_dataset(self, d):
        dir_imgs = os.path.join(d, "imgs")
        dir_masks = os.path.join(d, "masks")
        if os.path.isdir(dir_imgs) and os.path.isdir(dir_masks):
            return BasicDataset(dir_imgs, dir_masks, resize=self.resize)
        else:
            return None
        
    def next(self):
        d, self.counter = self._sequence_next(self.counter)
        if d is None:
            return None, None
        ds = self._get_dataset(d)
        return ds, d if ds is not None else self.next()
    
    def _sequence_next(self, counter):
        if self.counter < len(self.dir_path):
            return self.dir_path[self.counter], counter+1
        else:
            return None, counter
        
class CO3DDataSelector(iThorDataSelector): 
    def __init__(self, data_dir, strat="sequence", resize=None, shuffle_seed=None, insert_seen=True, limit_num=None):
        super().__init__(data_dir, strat=strat, resize=resize, shuffle_seed=shuffle_seed, insert_seen=insert_seen, limit_num=limit_num)
    
    def _get_obj_paths(self, shuffle_seed=None, insert_seen=True, limit_num=None):
        obj_types = os.listdir(self.data_dir)
        dir_names = []
        for obj in obj_types:
            dir_names += [os.path.join(obj, d) for d in sorted(os.listdir(os.path.join(self.data_dir, obj)))]
        dir_names = self._shuffle(dir_names, shuffle_seed)
        if insert_seen:
            dir_names = self._insert_seen_object(dir_names)
        dir_path = [os.path.join(self.data_dir, dn) for dn in dir_names if os.path.isdir(os.path.join(self.data_dir, dn))]
        if type(limit_num) == int and limit_num > 0:
            dir_path = dir_path[0:limit_num]
        print("[Data Selector] CO3D dataset, found object types: ", obj_types)
        return dir_path
    
    def _get_dataset(self, d):
        dir_imgs = os.path.join(d, "images")
        dir_masks = os.path.join(d, "masks")
        if os.path.isdir(dir_imgs) and os.path.isdir(dir_masks):
            return BasicDataset(dir_imgs, dir_masks, resize=self.resize, random_crop=True)
        else:
            print("[DataSelector Warning] found error dir: ", dir_imgs)
            return None
        
class DavisDataSelector(iThorDataSelector):
    def __init__(self, data_dir, strat="sequence", resize=None, shuffle_seed=None, insert_seen=True, limit_num=None):
        super().__init__(data_dir, strat=strat, resize=resize, shuffle_seed=shuffle_seed, insert_seen=insert_seen, limit_num=limit_num)
        
    def _get_obj_paths(self, shuffle_seed=None, insert_seen=True, limit_num=None):
        self.ImgPath = "JPEGImages"
        self.MaskPath = "Annotations"
        self.ResolutionPath = "480p"
        objPath = os.path.join(self.data_dir, self.ImgPath, self.ResolutionPath)
        objList = [obj for obj in sorted(os.listdir(objPath))]
        objList = self._shuffle(objList, shuffle_seed)
        if insert_seen:
            objList = self._insert_seen_object(objList)
        if type(limit_num) == int and limit_num > 0:
            objList = objList[0:limit_num]
        print("[Davis Data Selector] object list: ", objList)
        return objList

    def _get_dataset(self, d):
        dir_imgs = os.path.join(self.data_dir, self.ImgPath, self.ResolutionPath, d)
        dir_masks = os.path.join(self.data_dir, self.MaskPath, self.ResolutionPath, d)
        if os.path.isdir(dir_imgs) and os.path.isdir(dir_masks):
            return BasicDataset(dir_imgs, dir_masks, resize=self.resize, random_crop=True)
        else:
            print("[DataSelector Warning] found error dir: ", dir_imgs)
            return None