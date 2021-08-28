import os
import torch
from tqdm import tqdm

from model.coeffnet.coeffnet import Multinet

def genBases(checkpoint_path, output_dir, device=torch.device('cpu'), extension=".json"):
    """generate base files (z + corresponding output weights) based on trained Multinet

    Args:
        checkpoint_path (string): The path of checkpoint file of the Multinet, ends with .pth
        output_dir (string): The directory of output base files
        device (torch.device, optional): the device to put this operation on. Defaults to torch.device('cpu').
        extension (str, optional): the extension of the base file. Defaults to ".json".

    Raises:
        IOError: raised if the checkpoint file can't be found

    Returns:
        bool: True if all the base files successfully generated
    """
    # load checkpoint
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith('.pth'):
        state_dict = torch.load(checkpoint_path, map_location=device)
        zs = state_dict['z']
        base_num = zs.size(0)
        z_dim = zs.size(1)
    else:
        raise IOError
    
    # build Multinet
    net = Multinet(obj_num=base_num, z_dim=z_dim, device=device)
    net.load_state_dict(state_dict)
    net.to(device)
    hypernet = net.hypernet
    
    # prepare output
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    # generate bases file
    try:
        for i in tqdm(range(base_num)):
            input_z = zs[i]
            weights = hypernet(input_z)
            saved_file_path = os.path.join(output_dir, f"base_{i}{extension}")
            torch.save({'z':input_z, 'weights':weights}, saved_file_path)
    except Exception:
        return False
    else:
        return True
    
if __name__ == '__main__':
    if genBases("../checkpoints_conv_hypernet/checkpoint.pth", "./Bases/"):
        print("gen Bases success !")
    else:
        print("gen Bases fail !")

    