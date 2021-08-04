import os
import torch
import random
import argparse
from model.coeffnet.coeffnet import Coeffnet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, sampler
from eval import eval_net

def _get_args():
    parser = argparse.ArgumentParser(description='Evaluate hyper-coeffnet (object pursuit) performance on certain dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', dest='data', type=str, default='/data/pancy/iThor/single_obj/data_FloorPlan2_Kettle/',
                        help="data path of imgs and masks")
    parser.add_argument('-l', '--hyper', dest='hyper', type=str, default="./checkpoints_coeff_Kettle/Best.pth",
                        help='path for hypernet parameter')
    parser.add_argument('-z', '--z', dest='z', type=str, default="./Bases/Plate.json",
                        help='path for z')
    parser.add_argument('-n', '--size', dest='data_size', type=int, default=400,
                        help='Size of testing data')

    return parser.parse_args()

if __name__ == "__main__":
    # config
    args = _get_args()
    hypernet_param_path = args.hyper
    z_path = args.z
    data_dir = args.data
    data_size = args.data_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Coeffnet(z_dim=100, device=device)
    net.to(device=device)
    net.load_state_dict(torch.load(hypernet_param_path, map_location=device))
    net.load_z(z_path)
    print(f"load hypernet param from {hypernet_param_path} \n"
          f"load z from {z_path} \n"
          f"device: {device} \n"
          f"data dir: {data_dir} \n"
          f"eval data size: {data_size}\n")
    
    # dataset
    img_dir = os.path.join(data_dir, "imgs")
    mask_dir = os.path.join(data_dir, "masks")
    eval_dataset = BasicDataset(img_dir, mask_dir, 1)

    n_size = len(eval_dataset)
    indices = [i for i in range(n_size)]
    random.shuffle(indices)
    if data_size is not None:
        eval_sampler = sampler.SubsetRandomSampler(indices[:min(n_size, data_size)])
    else:
        eval_sampler = sampler.SubsetRandomSampler(indices[:n_size])
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=True, sampler=eval_sampler)
    
    res, _ = eval_net(net, eval_loader, device)
    print("average dice coeff: ", res)