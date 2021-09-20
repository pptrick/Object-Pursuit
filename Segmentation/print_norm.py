import torch
from object_pursuit.pursuit import get_z_bases

zs = get_z_bases(100, './checkpoints_objectpursuit_rhino_yes_l1norm_coeff/zs/', torch.device('cpu'))
for i,z in enumerate(zs):
    print(f"No.{i} z, norm: {torch.norm(z)}")
