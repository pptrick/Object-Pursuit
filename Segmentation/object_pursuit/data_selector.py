import os
import copy
import random

from dataset.basic_dataset import BasicDataset
from dataset.davis_dataset import DavisDataset

class iThorDataSelector(object):    
    def __init__(self, data_dir, strat="sequence", resize=None, shuffle_seed=None, insert_seen=True):
        assert os.path.isdir(data_dir)
        self.strat = strat
        self.resize = resize
        self.dir_names = sorted(os.listdir(data_dir))
        if shuffle_seed is not None and strat == "sequence":
            r = random.random
            random.seed(shuffle_seed)
            random.shuffle(self.dir_names, random=r)
        elif strat == "random":
            random.shuffle(self.dir_names)
        if insert_seen:
            self.dir_names = self._insert_seen_object(self.dir_names)
        print("[iThor Data Selector] object list: ", self.dir_names)
        self.dir_path = [os.path.join(data_dir, dn) for dn in self.dir_names]
        self.counter = 0
    
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
        
class DavisDataSelector(object):
    def __init__(self, data_dir, strat="sequence", resize=None):
        # self.objects, _, _, _ = DavisDataset.get_obj_list(data_dir)
        self.objects = ['rhino', 'bear', 'dog', 'blackswan', 'cows', 'bus']
        self.dataset_dir = data_dir
        self.strat = strat
        self.resize = resize
        self.counter = 0
        self.remain_set = copy.deepcopy(self.objects)
        
    def _get_dataset(self, d):
        return DavisDataset(self.dataset_dir, d, self.resize)
        
    def next(self):
        if self.strat == "sequence":
            d, self.counter = self._sequence_next(self.counter)
            if d is None:
                return None, None
            ds = self._get_dataset(d)
            return ds, d if ds is not None else self.next()
        elif self.strat == "random":
            d, self.remain_set = self._random_next(self.remain_set)
            if d is None:
                return None, None
            ds = self._get_dataset(d)
            return ds, d if ds is not None else self.next()
        else:
            raise NotImplementedError
        
    def _sequence_next(self, counter):
        if self.counter < len(self.objects):
            return self.objects[self.counter], counter+1
        else:
            return None, counter
        
    def _random_next(self, remain_set):
        if len(remain_set) > 0:
            d = random.choice(remain_set)
            remain_set.remove(d)
            return d, remain_set
        else:
            return None, []