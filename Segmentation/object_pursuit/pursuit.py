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
            
def should_retrain(max_val_acc):
    if max_val_acc < 0.85:
        return True
    else:
        return False

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
    create_dir(os.path.join(output_dir, "explored_objects"))
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    create_dir(checkpoint_dir)
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
    
    # pursuit info
    pursuit_info = f'''Starting pursuing:
        z_dim:                  {z_dim}
        object data dir:        {data_dir}
        output dir:             {output_dir}
        device:                 {device}
        pretrained hypernet:    {pretrained_hypernet}
        pretrained backbone:    {pretrained_backbone}
        pretrained bases:       {pretrained_bases}
        data select strategy:   {select_strat}
        data resize:            {resize}
        bases dir:              {base_dir}
    '''
    write_log(log_file, pursuit_info)
    
    new_obj_dataset, obj_data_dir = dataSelector.next()
    obj_counter = 0
    while new_obj_dataset is not None:
        zs = get_z_bases(z_dim, base_dir, device)
        base_num = len(zs)
        
        # for each new object, create a new dir
        obj_dir = os.path.join(output_dir, "explored_objects", f"obj_{obj_counter}")
        create_dir(obj_dir)
        
        new_obj_info = f'''Starting new object:
            current base num:    {base_num}
            object data dir:     {obj_data_dir}
            object index:        {obj_counter}
            output obj dir:      {obj_dir}
        '''
        max_val_acc = 0
        write_log(log_file, "\n===============start new object================")
        write_log(log_file, new_obj_info)
        
        # test if a new object can be expressed by other objects
        if base_num > 0:
            write_log(log_file, "start coefficient pursuit:")
            # freeze the hypernet and backbone
            freeze(hypernet=hypernet, backbone=backbone)
            coeff_pursuit_dir = os.path.join(obj_dir, "coeff_pursuit")
            create_dir(coeff_pursuit_dir)
            write_log(log_file, f"coeff pursuit result dir: {coeff_pursuit_dir}")
            max_val_acc = train_net(z_dim=z_dim, base_num=base_num, dataset=new_obj_dataset, device=device,
                      zs=zs, 
                      net_type="coeffnet", 
                      hypernet=hypernet, 
                      backbone=backbone,
                      save_cp_path=coeff_pursuit_dir,
                      base_dir=base_dir,
                      max_epochs=80,
                      lr=8e-5)
            write_log(log_file, f"training stop, max validation acc: {max_val_acc}")
        # if not, train this object as a new base
        if should_retrain(max_val_acc): # the condition to retrain a new base
            write_log(log_file, "start to train as new base:")
            # unfreeze the backbone
            unfreeze(hypernet=hypernet)
            base_update_dir = os.path.join(obj_dir, "base_update")
            create_dir(base_update_dir)
            write_log(log_file, f"base update result dir: {base_update_dir}")
            max_val_acc = train_net(z_dim=z_dim, base_num=base_num, dataset=new_obj_dataset, device=device,
                      net_type="singlenet",
                      hypernet=hypernet,
                      backbone=backbone,
                      save_cp_path=base_update_dir,
                      base_dir=base_dir,
                      max_epochs=80,
                      lr=8e-5)
            write_log(log_file, f"training stop, max validation acc: {max_val_acc}")
        new_obj_dataset, obj_data_dir = dataSelector.next()
        obj_counter += 1
        # save checkpoint
        torch.save(hypernet.state_dict(), os.path.join(checkpoint_dir, f'hypernet.pth'))
        torch.save(backbone.state_dict(), os.path.join(checkpoint_dir, f'backbone.pth'))
        write_log(log_file, "\n===============end object================")
        
    log_file.close()
    
    
    
        
    