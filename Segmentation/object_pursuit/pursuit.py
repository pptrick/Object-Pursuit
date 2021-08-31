import os
import torch

from model.coeffnet.hypernet import Hypernet
from model.coeffnet.coeffnet_simple import Backbone
from model.coeffnet.coeffnet_simple import init_backbone, init_hypernet
from object_pursuit.data_selector import iThorDataSelector
from utils.GenBases import genBases
from object_pursuit.train import train_net, create_dir, write_log

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
    
def freeze(hypernet=None, backbone=None):
    if hypernet is not None:
        for param in hypernet.parameters():
            param.requires_grad = False
    if backbone is not None:
        for param in backbone.parameters():
            param.requires_grad = False
            
def unfreeze(hypernet=None, backbone=None):
    if hypernet is not None:
        for param in hypernet.parameters():
            param.requires_grad = True
    if backbone is not None:
        for param in backbone.parameters():
            param.requires_grad = True

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
    create_dir(output_dir)
    base_dir = os.path.join(output_dir, "Bases")
    create_dir(base_dir)
    create_dir(os.path.join(output_dir, "obj"))
    log_file = open(os.path.join(output_dir, "pursuit_log.txt"), "w")
    
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
    dataSelector = iThorDataSelector(data_dir, strat=select_strat, resize=resize)
    
    new_obj_dataset, obj_data_dir = dataSelector.next()
    obj_counter = 0
    while new_obj_dataset is not None:
        zs = get_z_bases(z_dim, base_dir, device)
        base_num = len(zs)
        
        # for each new object, create a new dir
        obj_dir = os.path.join(output_dir, "obj", f"obj_{obj_counter}")
        create_dir(obj_dir)
        
        # TODO: test if a new object can be expressed by other objects
        if base_num > 0:
            # freeze the hypernet and backbone
            freeze(hypernet=hypernet, backbone=backbone)
            coeff_pursuit_dir = os.path.join(obj_dir, "coeff_pursuit")
            create_dir(coeff_pursuit_dir)
            train_net(z_dim=z_dim, base_num=base_num, dataset=new_obj_dataset, device=device,
                      zs=zs, 
                      net_type="coeffnet", 
                      hypernet=hypernet, 
                      backbone=backbone,
                      save_cp_path=coeff_pursuit_dir,
                      base_dir=base_dir,
                      epochs=1)
        # TODO: if not, train this object as a new base
        if True: # TODO: the condition to retrain a new base
            # unfreeze the backbone
            unfreeze(hypernet=hypernet)
            base_update_dir = os.path.join(obj_dir, "base_update")
            create_dir(base_update_dir)
            train_net(z_dim=z_dim, base_num=base_num, dataset=new_obj_dataset, device=device,
                      net_type="singlenet",
                      hypernet=hypernet,
                      backbone=backbone,
                      save_cp_path=base_update_dir,
                      base_dir=base_dir,
                      epochs=1)
        new_obj_dataset, obj_data_dir = dataSelector.next()
        obj_counter += 1
        
    log_file.close()
    
    
    
        
    