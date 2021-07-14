import os
import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

class ColorCounter(object):
    def __init__(self, img_dir, mask_dir):
        b, g, r = self._calHistinDir(img_dir, mask_dir)
        self._PlotHist(b, g, r)
    
    def _calHist(self, img_path, mask_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        b_hist = np.array(cv2.calcHist([img], [0], mask, [256], [0, 256]))
        g_hist = np.array(cv2.calcHist([img], [1], mask, [256], [0, 256]))
        r_hist = np.array(cv2.calcHist([img], [2], mask, [256], [0, 256]))
        return b_hist, g_hist, r_hist
    
    def _calHistinDir(self, img_dir, mask_dir):
        #check valid
        try:
            assert(os.path.isdir(img_dir) and os.path.isdir(mask_dir))
            self.img_mask = []
            img_list = os.listdir(img_dir)
            for img in img_list:
                img_file = os.path.join(img_dir, img)
                mask_file = os.path.join(mask_dir, img)
                assert(os.path.isfile(img_file) and os.path.isfile(mask_file))
                self.img_mask.append((img_file, mask_file))
        except Exception as e:
            print("[Error] fail to cal Hist: ", e)
            exit(-1)
        else:
            b_hist, g_hist, r_hist = np.zeros((256, 1)), np.zeros((256, 1)), np.zeros((256, 1))
            for im in tqdm.tqdm(self.img_mask):
                b, g, r = self._calHist(im[0], im[1])
                b_hist += b
                r_hist += r
                g_hist += g
            return b_hist, g_hist, r_hist
    
    def _PlotHist(self, b_hist, g_hist, r_hist):
        plt.plot(b_hist, label='B', color='blue')
        plt.plot(g_hist, label='G', color='green')
        plt.plot(r_hist, label='R', color='red')
        plt.legend(loc='best')
        plt.xlim([0, 256])
        plt.savefig("Hist_Plate2.png")
        
if __name__ == "__main__":
    cc = ColorCounter('/data/pancy/iThor/single_obj/data_FloorPlan2_Plate/imgs', '/data/pancy/iThor/single_obj/data_FloorPlan2_Plate/masks')
    # cc = ColorCounter('/home/pancy/IP/ithor/DataGen/data_FloorPlan3_Plate/imgs', '/home/pancy/IP/ithor/DataGen/data_FloorPlan3_Plate/masks')