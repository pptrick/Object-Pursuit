import json
import re
import os

def BaseRecorder(joint_log, pursuit_log, save_dir):
    base_info = []
    # add joint origin
    joint_fp = open(joint_log, 'r')
    origin_base_info = json.load(joint_fp)
    for inf in origin_base_info:
        base_info.append({'data_dir': inf['data_dir'], 'index': inf['index']})
    pursuit_fp = open(pursuit_log)
    log = pursuit_fp.readlines()
    obj_data_path = None
    for line in log:
        res_dir = re.match(r'.*object data dir:     (.*)', line)
        res_add = re.match(r'.*add \'base_([0-9]+).json\' to bases', line)
        if res_dir is not None:
            obj_data_path = res_dir.group(1)
        if res_add is not None:
            base_index = res_add.group(1)
            base_info.append({'data_dir': obj_data_path, 'index': int(base_index)})
    
    with open(os.path.join(save_dir, "base_info.json"), 'w') as fp:
        json.dump(base_info, fp)
    
    
if __name__ == "__main__":
    BaseRecorder("/orion/u/pancy/project/Object-Pursuit/Segmentation/checkpoints_simple_zs/base_info_origin.json",
                 "/orion/u/pancy/project/Object-Pursuit/Segmentation/checkpoints_sequence_threshold_0.8/pursuit_log.txt",
                 "/orion/u/pancy/project/Object-Pursuit/Segmentation/checkpoints_sequence_threshold_0.8/")