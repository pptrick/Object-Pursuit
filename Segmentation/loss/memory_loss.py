import os
import random
import torch
import torch.nn as nn


class MemoryLoss(nn.Module):
    def __init__(self, Base_dir, device):
        super(MemoryLoss, self).__init__()
        assert(os.path.isdir(Base_dir))
        self.Base_dir = Base_dir
        self.device = device
        self.file_list = [os.path.join(Base_dir, file) for file in os.listdir(Base_dir) if file.endswith(".json")]
        # preload
        self._preload(self.file_list)
        
    def _preload(self, file_list):
        print("preload from ", file_list)
        self.z = []
        self.weights = []
        for file in file_list:
            records = torch.load(file, map_location=self.device)
            self.z.append(records['z'])
            self.weights.append(records['weights'])
            
    def _l2_loss(self, pred, gt):
        loss = None
        for param in gt:
            if loss is None:
                loss = torch.norm(pred[param]-gt[param])
            else:
                loss += torch.norm(pred[param]-gt[param])
        return loss
            
    def forward(self, hypernet, mem_coeff):
        loss = None
        # TODO: consider change loss cal to randomly/sequential choice
        index_list = range(len(self.z))
        index_list = random.sample(index_list, min(10, len(index_list)))
        for i in index_list:
            pred_w = hypernet(self.z[i])
            gt_w = self.weights[i]
            # if loss is None:
            #     loss = self._l2_loss(pred_w, gt_w)
            # else:
            #     loss += self._l2_loss(pred_w, gt_w)
            loss = mem_coeff * self._l2_loss(pred_w, gt_w)
            loss.backward()
        return loss
            