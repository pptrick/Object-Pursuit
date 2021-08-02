import os
import torch
import numpy as np

def _neglect(param, prefix_neglect, suffix_neglect):
    assert(isinstance(param, str))
    for prefix in prefix_neglect:
        if param.startswith(prefix):
            return True
    for suffix in suffix_neglect:
        if param.endswith(suffix):
            return True
    return False

def weight2vec(weight_file, prefix_neglect=[], suffix_neglect=[]):
    state_dict = torch.load(weight_file, map_location='cpu')
    vec_list = []
    for param in state_dict:
        if not _neglect(param, prefix_neglect, suffix_neglect):
            vec = state_dict[param].numpy().ravel()
            vec_list.append(vec)
    
    return np.concatenate(vec_list)
    

def save_vec(weight_vec, dir, name="weight"):
    np.save(os.path.join(dir, name), weight_vec)
    
def viz_weight(weight_file):
    state_dict = torch.load(weight_file, map_location='cpu')
    # for param in state_dict:
    #     print(param)
    for param in state_dict['weights']:
        print(param, state_dict['weights'][param].size())
    print(state_dict['z'])


if __name__ == "__main__":
    # obj = "Pot"
    # name = "fp2_"+obj
    # weight_vec = weight2vec(f"/data/pancy/object_pursuit/checkpoints/pretrained_backbone/checkpoints_{obj}_FP2/MODEL_{obj}_fp2.pth", ["backbone"], ["running_mean", "running_var", "num_batches_tracked"])
    # weight_vec = weight2vec(f"../Segmentation/INTERRUPTED.pth", ["backbone"], ["running_mean", "running_var", "num_batches_tracked"])
    # save_vec(weight_vec, dir="./Vec", name="Random")
    # print(weight_vec.shape)
    viz_weight("../Segmentation/Bases/Cup.json")