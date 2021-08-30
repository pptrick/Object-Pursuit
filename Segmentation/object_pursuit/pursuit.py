import os
import torch

from model.coeffnet.hypernet import Hypernet
from model.coeffnet.coeffnet_simple import Backbone
from model.coeffnet.coeffnet_simple import init_backbone, init_hypernet
from object_pursuit.data_selector import iThorDataSelector
from utils.GenBases import genBases
from object_pursuit.train import train_net

def get_z_bases(z_dim, base_path, device):
    if os.path.isdir(base_path):
        base_files = [os.path.join(base_path, file) for file in sorted(os.listdir(base_path)) if file.endswith(".json")]
        zs = []
        for f in base_files:
            z = torch.load(f, map_location=device)['z']
            assert(z.size()[0] == z_dim)
            zs.append(z)
        return zs
    elif os.path.isfile(base_path):
        assert(os.path.isfile(base_path) and base_path.endswith(".pth"))
        zs = []
        zs_sync = torch.load(base_path, map_location=device)['z']
        for i in range(len(zs_sync)):
            zs.append(zs_sync[i].to(device))
        return zs
    else:
        raise IOError

def pursuit(z_dim, 
            data_dir, 
            output_dir, 
            device, 
            pretrained_bases=None,
            pretrained_backbone=None, 
            pretrained_hypernet=None, 
            select_strat="sequence", 
            resize=None):
    # prepare for new pursuit dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    base_dir = os.path.join(output_dir, "Bases")
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    
    # prepare bases
    if pretrained_bases is not None:
        genBases(pretrained_bases, base_dir, device=device)
    
    # build hypernet
    hypernet = Hypernet(z_dim)
    if pretrained_hypernet is not None:
        init_hypernet(pretrained_hypernet, hypernet, device)
    hypernet.to(device)
    
    # build backbone
    backbone = Backbone()
    if pretrained_backbone is not None:
        init_backbone(pretrained_backbone, backbone, device, freeze=True)
    backbone.to(device)
    
    # data selector
    dataSelector = iThorDataSelector(data_dir, strat=select_strat, resize=None)
    
    new_obj_dataset = dataSelector.next()
    while new_obj_dataset is not None:
        # TODO: test if a new object can be expressed by other objects
        zs = get_z_bases(z_dim, base_dir, device)
        base_num = len(zs)
        if base_num > 0:
            train_net(z_dim=z_dim, base_num=base_num, dataset=new_obj_dataset, device=device,
                      zs=zs, 
                      net_type="coeffnet", 
                      hypernet=hypernet, 
                      backbone=backbone,
                      save_cp_path=None, #TODO
                      base_dir=base_dir,
                      epochs=1)
        # TODO: if not, retrain this object
        if True: # TODO: the condition to retrain a new base
            train_net(z_dim=z_dim, base_num=base_num, dataset=new_obj_dataset, device=device,
                      net_type="singlenet",
                      hypernet=hypernet,
                      backbone=backbone,
                      save_cp_path=None, #TODO
                      base_dir=base_dir,
                      epochs=1)
        new_obj_dataset = dataSelector.next()
    
    
    
        
    