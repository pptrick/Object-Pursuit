import os
import copy
import random
from dataset.basic_dataset import BasicDataset
from dataset.davis_dataset import DavisDataset

class iThorDataSelector(object):
    def __init__(self, data_dir, strat="sequence", resize=None):
        assert os.path.isdir(data_dir)
        self.strat = strat
        self.resize = resize
        self.dir_names = os.listdir(data_dir)
        self.dir_path = [os.path.join(data_dir, dn) for dn in self.dir_names]
        self.counter = 0
        self.remain_set = copy.deepcopy(self.dir_path)
        
    def _get_dataset(self, d):
        dir_imgs = os.path.join(d, "imgs")
        dir_masks = os.path.join(d, "masks")
        if os.path.isdir(dir_imgs) and os.path.isdir(dir_masks):
            return BasicDataset(dir_imgs, dir_masks, resize=self.resize)
        else:
            return None
        
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
        if self.counter < len(self.dir_path):
            return self.dir_path[self.counter], counter+1
        else:
            return None, counter
        
    def _random_next(self, remain_set):
        if len(remain_set) > 0:
            d = random.choice(remain_set)
            remain_set.remove(d)
            return d, remain_set
        else:
            return None, []
        
class DavisDataSelector(object):
    def __init__(self, data_dir, strat="sequence", resize=None):
        # self.objects, _, _, _ = DavisDataset.get_obj_list(data_dir)
        self.objects = ['bike-packing', 'rhino', 'bear', 'dog', 'blackswan', 'cows', 'bus']
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