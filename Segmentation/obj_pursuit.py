import torch
from model.coeffnet.coeffnet_deeplab import Coeffnet_Deeplab

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
net = Coeffnet_Deeplab("/home/pancy/IP/Object-Pursuit/Segmentation/Bases", device)
net.to(device=device)
for i in net.parameters():
    if i.requires_grad:
        print(i)
