# select dataset for n-shot learning
import torch
from torch.utils.data import random_split
from dataset.basic_dataset import BasicDataset, BasicDataset_nshot

def select_dataset(dataset,
                   img_dir,
                   mask_dir,
                   resize=None,
                   n_test=None,
                   n=1,
                   shuffle_seed=3,
                   split_seed=0):
    if dataset == "iThor":
        test_dataset = BasicDataset(img_dir, mask_dir, resize=resize)
        if n>0:
            train_dataset = BasicDataset_nshot(img_dir, mask_dir, n=n, resize=resize, shuffle_seed=shuffle_seed)
        else:
            train_dataset = BasicDataset(img_dir, mask_dir, resize=resize)
    elif dataset == "DAVIS":
        # here I crop DAVIS' 854*480 image to a square then resize
        test_dataset = BasicDataset(img_dir, mask_dir, resize=resize, random_crop=True)
        if n>0:
            train_dataset = BasicDataset_nshot(img_dir, mask_dir, n=n, resize=resize, random_crop=True)
        else:
            train_dataset = BasicDataset(img_dir, mask_dir, resize=resize, random_crop=True)
    elif dataset == "KITTI":
        pass
    else:
        raise NotImplementedError
    test_dataset = split_testset(test_dataset, n_test, split_seed)
    return train_dataset, test_dataset

def split_testset(test_dataset, n_test=None, split_seed=0):
    # split out a sub-dataset from original testset, do it when original testset is large
    if type(n_test) == int and n_test < len(test_dataset) and n_test > 0:
        new_test_dataset, _ = random_split(test_dataset, [n_test, len(test_dataset) - n_test], generator=torch.Generator().manual_seed(split_seed))
        return new_test_dataset
    else:
        return test_dataset