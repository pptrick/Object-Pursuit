import os
import torch
import random
from model.coeffnet.coeffnet import Singlenet
from dataset.basic_dataset import BasicDataset
from torch.utils.data import DataLoader, sampler
from evaluation.eval_net import eval_net

def test_seen_obj(data_dir, z_dir, hypernet_path, backbone_path, test_size=400, threshold=0.7):
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    net = Singlenet(z_dim=100, device=device)
    net.to(device=device)
    net.init_hypernet(hypernet_path)
    net.init_backbone(backbone_path)
    
    z_files = [os.path.join(z_dir, zf) for zf in sorted(os.listdir(z_dir)) if zf.endswith('.json')]
    img_dir = os.path.join(data_dir, "imgs")
    mask_dir = os.path.join(data_dir, "masks")
    eval_dataset = BasicDataset(img_dir, mask_dir, (256, 256))
    
    n_size = len(eval_dataset)
    indices = [i for i in range(n_size)]
    random.shuffle(indices)
    if test_size is not None:
        eval_sampler = sampler.SubsetRandomSampler(indices[:min(n_size, test_size)])
    else:
        eval_sampler = sampler.SubsetRandomSampler(indices[:n_size])
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=True, sampler=eval_sampler)
    
    max_acc = threshold
    argmax_zf = None
    for zf in z_files:
        net.load_z(zf)
        res, _ = eval_net(net, eval_loader, device)
        print(f"z file: {zf}, test accuracy: {res}")
        if res >= max_acc:
            max_acc = res
            argmax_zf = zf
    print(f"The argmax z file is {argmax_zf}, max acc: {max_acc}")
        