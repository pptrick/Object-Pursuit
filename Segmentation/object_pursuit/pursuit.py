from itertools import count
import os
import torch
import shutil
import json

from tqdm import tqdm
from model.coeffnet.hypernet import Hypernet
from model.coeffnet.coeffnet_simple import Backbone
from model.coeffnet.coeffnet_simple import init_backbone, init_hypernet
from object_pursuit.data_selector import iThorDataSelector, DavisDataSelector
from utils.GenBases import genBases
from object_pursuit.train import train_net, have_seen
from utils.util import *

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
            z = zs_sync[i].to(device)
            assert(z.size()[0] == z_dim)
            zs.append(z)
        return zs
    else:
        raise IOError
    
def save_base_as_init_objects(bases, z_dir, hypernet=None):
    if os.path.isdir(z_dir):
        for i,z in enumerate(tqdm(bases)):
            file_path = os.path.join(z_dir, f"z_{'%04d' % i}.json")
            if hypernet is not None:
                weights = hypernet(z)
                torch.save({'z':z, 'weights':weights}, file_path)
            else:
                torch.save({'z':z}, file_path)
                
def copy_zs(src_dir, target_dir):
    create_dir(target_dir)
    if os.path.isdir(src_dir) and os.path.isdir(target_dir):
        files = os.listdir(src_dir)
        for f in files:
            if f.endswith(".json"):
                path = os.path.join(src_dir, f)
                z = torch.load(path, map_location=torch.device('cpu'))['z']
                torch.save({'z':z}, os.path.join(target_dir, f))
            
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
            
def can_be_expressed(max_val_acc, threshold):
    if max_val_acc >= threshold:
        return True
    else:
        return False
    
def least_square(bases, target):
    tar = torch.unsqueeze(target, dim=-1)
    A = torch.stack(bases, dim=1)
    coeff_mat = torch.mm(A.T, A)
    proj = torch.mm(A.T, tar)
    coeff = torch.mm(torch.inverse(coeff_mat), proj)
    res = torch.mm(A, coeff)
    res = torch.squeeze(res)
    coeff = torch.squeeze(coeff)
    # distance
    dist = torch.norm(target-res)/torch.norm(target)
    return res, coeff, dist

def pursuit(z_dim, 
            data_dir, 
            output_dir, 
            device, 
            initial_zs = None,
            pretrained_bases=None,
            pretrained_backbone=None, 
            pretrained_hypernet=None, 
            select_strat="sequence", 
            resize=None,
            express_threshold=0.7,
            log_info="default"):
    # prepare for new pursuit dir
    create_dir(output_dir)
    base_dir = os.path.join(output_dir, "Bases")
    create_dir(base_dir)
    z_dir = os.path.join(output_dir, "zs")
    create_dir(z_dir)
    create_dir(os.path.join(output_dir, "explored_objects"))
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    create_dir(checkpoint_dir)
    log_file = open(os.path.join(output_dir, "pursuit_log.txt"), "w")
    write_log(log_file, "[Exp Info] "+log_info)
    z_info = []
    
    # prepare bases
    if pretrained_bases is not None and os.path.isfile(pretrained_bases):
        genBases(pretrained_bases, base_dir, device=device)
    elif pretrained_bases is not None and os.path.isdir(pretrained_bases):
        base_files = [os.path.join(pretrained_bases, file) for file in sorted(os.listdir(pretrained_bases)) if file.endswith(".json")]
        for f in base_files:
            shutil.copy(f, base_dir)
    
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
    dataSelector = iThorDataSelector(data_dir, strat=select_strat, resize=resize, shuffle_seed=1)
    # dataSelector = DavisDataSelector(data_dir, strat=select_strat, resize=resize)
    
    # initialize bases
    init_bases = get_z_bases(z_dim, base_dir, device)
    init_base_num = len(init_bases)
    
    # initialize current object list
    if initial_zs is None:
        initial_zs = base_dir
    init_objects = get_z_bases(z_dim, initial_zs, device)
    init_objects_num = len(init_objects)
    save_base_as_init_objects(init_objects, z_dir, hypernet=hypernet)
    obj_counter = init_objects_num
    
    # pursuit info
    pursuit_info = f'''Starting pursuing:
        z_dim:                            {z_dim}
        object data dir:                  {data_dir}
        output dir:                       {output_dir}
        device:                           {device}
        pretrained hypernet:              {pretrained_hypernet}
        pretrained backbone:              {pretrained_backbone}
        pretrained bases:                 {pretrained_bases}
        data select strategy:             {select_strat}
        data resize:                      {resize}
        bases dir:                        {base_dir}
        pretrained (initial) base index:  0~{init_base_num-1}  
        initial (first) object index:     {obj_counter}
        express accuracy threshold:       {express_threshold}
    '''
    write_log(log_file, pursuit_info)
    
    new_obj_dataset, obj_data_dir = dataSelector.next()
    counter = 0
    
    while new_obj_dataset is not None:
        bases = get_z_bases(z_dim, base_dir, device)
        base_num = len(bases)
        
        # record checkpoints per 8 round
        counter += 1
        if counter % 8 == 0:
            temp_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_round_{counter}")
            create_dir(temp_checkpoint_dir)
            torch.save(hypernet.state_dict(), os.path.join(temp_checkpoint_dir, f'hypernet.pth'))
            copy_zs(z_dir, os.path.join(temp_checkpoint_dir, "zs"))
            copy_zs(base_dir, os.path.join(temp_checkpoint_dir, "Bases"))
            write_log(log_file, f"[checkpoint] pursuit round {counter} has been saved to {temp_checkpoint_dir}")
        
        # for each new object, create a new dir
        obj_dir = os.path.join(output_dir, "explored_objects", f"obj_{obj_counter}")
        create_dir(obj_dir)
        
        new_obj_info = f'''Starting new object:
            round:               {counter}
            current base num:    {base_num}
            object data dir:     {obj_data_dir}
            object index:        {obj_counter}
            output obj dir:      {obj_dir}
        '''
        max_val_acc = 0.0
        write_log(log_file, "\n=============================start new object==============================")
        write_log(log_file, new_obj_info)
        
        # ========================================================================================================
        # check if current object has been seen
        seen, acc, z_file = have_seen(new_obj_dataset, device, z_dir, z_dim, hypernet, backbone, express_threshold, start_index=init_objects_num)
        if seen:
            write_log(log_file, f"Current object has been seen! corresponding z file: {z_file}, express accuracy: {acc}")
            new_obj_dataset, obj_data_dir = dataSelector.next()
            shutil.rmtree(obj_dir)
            write_log(log_file, "\n===============================end object==================================")
            continue
        else:
            write_log(log_file, f"Current object is novel, max acc: {acc}, most similiar object: {z_file}, start object pursuit")
        
        # ========================================================================================================
        # (first check) test if a new object can be expressed by other objects
        if base_num > 0:
            write_log(log_file, "start coefficient pursuit (first check):")
            # freeze the hypernet and backbone
            freeze(hypernet=hypernet, backbone=backbone)
            coeff_pursuit_dir = os.path.join(obj_dir, "coeff_pursuit")
            create_dir(coeff_pursuit_dir)
            write_log(log_file, f"coeff pursuit result dir: {coeff_pursuit_dir}")
            max_val_acc, coeff_net = train_net(z_dim=z_dim, base_num=base_num, dataset=new_obj_dataset, device=device,
                      zs=bases, 
                      net_type="coeffnet", 
                      hypernet=hypernet, 
                      backbone=backbone,
                      save_cp_path=coeff_pursuit_dir,
                      z_dir=z_dir,
                      max_epochs=200,
                      lr=1e-4,
                      l1_loss_coeff=0.2)
            write_log(log_file, f"training stop, max validation acc: {max_val_acc}")
        # ==========================================================================================================
        # (train as a new base) if not, train this object as a new base
        if not can_be_expressed(max_val_acc, express_threshold): # the condition to retrain a new base
            write_log(log_file, "can't be expressed by bases, start to train as new base:")
            # unfreeze the backbone
            unfreeze(hypernet=hypernet)
            base_update_dir = os.path.join(obj_dir, "base_update")
            create_dir(base_update_dir)
            write_log(log_file, f"base update result dir: {base_update_dir}")
            max_val_acc, z_net = train_net(z_dim=z_dim, base_num=base_num, dataset=new_obj_dataset, device=device,
                      net_type="singlenet",
                      hypernet=hypernet,
                      backbone=backbone,
                      save_cp_path=base_update_dir,
                      z_dir=z_dir,
                      max_epochs=200,
                      wait_epochs=5,
                      lr=1e-4,
                      l1_loss_coeff=0.1,
                      mem_loss_coeff=0.04)
            write_log(log_file, f"training stop, max validation acc: {max_val_acc}")
            
            # if the object is invalid
            if max_val_acc < express_threshold:
                write_log(log_file, f"[Warning] current object (data path: {obj_data_dir}) is invalid! The validation acc should be at least {express_threshold}, current acc {max_val_acc}; All records will be removed !")
                # reset backbone and hypernet
                init_backbone(os.path.join(checkpoint_dir, f'backbone.pth'), backbone, device, freeze=True)
                init_hypernet(os.path.join(checkpoint_dir, f'hypernet.pth'), hypernet, device, freeze=True)
                new_obj_dataset, obj_data_dir = dataSelector.next()
                shutil.rmtree(obj_dir)
                write_log(log_file, "\n===============================end object=================================")
                continue
            
            # ======================================================================================================
            # (second check) check new z can now be approximated (expressed by coeffs) by current bases
            if base_num > 0:
                write_log(log_file, f"start to examine whether the object {obj_counter} can be expressed by bases now (second check):")
                # freeze the hypernet and backbone
                freeze(hypernet=hypernet, backbone=backbone)
                check_express_dir = os.path.join(obj_dir, "check_express")
                create_dir(check_express_dir)
                write_log(log_file, f"check express result dir: {check_express_dir}")
                max_val_acc, examine_coeff_net = train_net(z_dim=z_dim, base_num=base_num, dataset=new_obj_dataset, device=device,
                        zs=bases, 
                        net_type="coeffnet", 
                        hypernet=hypernet, 
                        backbone=backbone,
                        save_cp_path=check_express_dir,
                        z_dir=z_dir,
                        max_epochs=200,
                        wait_epochs=5,
                        lr=1e-4,
                        acc_threshold=1.0,
                        l1_loss_coeff=0.2)
            else:
                max_val_acc = 0.0
            
            if can_be_expressed(max_val_acc, express_threshold):
                write_log(log_file, f"new z can be expressed by current bases, max val acc: {max_val_acc}, don't add it to bases")
                # save object's z
                write_log(log_file, f"object {obj_counter} pursuit complete, save object z 'z_{'%04d' % obj_counter}.json' to {z_dir}")
                examine_coeff_net.save_z(os.path.join(z_dir, f"z_{'%04d' % obj_counter}.json"), bases, hypernet)
            else:
                # save z as a new base
                write_log(log_file, f"new z can't be expressed by current bases, express max val acc: {max_val_acc}, add 'base_{'%04d' % base_num}.json' to bases")
                z_net.save_z(os.path.join(base_dir, f"base_{'%04d' % base_num}.json"), hypernet)
                # save object's z
                write_log(log_file, f"object {obj_counter} pursuit complete, save object z 'z_{'%04d' % obj_counter}.json' to {z_dir}")   
                z_net.save_z(os.path.join(z_dir, f"z_{'%04d' % obj_counter}.json"), hypernet)
            # ======================================================================================================
            
        else:
            # save object's z
            write_log(log_file, f"object {obj_counter} pursuit complete, save object z 'z_{'%04d' % obj_counter}.json' to {z_dir}")    
            coeff_net.save_z(os.path.join(z_dir, f"z_{'%04d' % obj_counter}.json"), bases, hypernet)
            
        z_info.append({
            "index": obj_counter,
            "data_dir": obj_data_dir,
            "z_file": f"z_{'%04d' % obj_counter}.json"
        })
        with open(os.path.join(output_dir, "z_info.json"), "w") as f:
            json.dump(z_info, f)
        
        write_log(log_file, f"save hypernet and backbone to {checkpoint_dir}, move to next object")     
        new_obj_dataset, obj_data_dir = dataSelector.next()
        obj_counter += 1
        # save checkpoint
        torch.save(hypernet.state_dict(), os.path.join(checkpoint_dir, f'hypernet.pth'))
        torch.save(backbone.state_dict(), os.path.join(checkpoint_dir, f'backbone.pth'))
        write_log(log_file, "\n===============================end object=================================")
        
    log_file.close()
    
    
    
        
    