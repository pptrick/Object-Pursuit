import os
from sklearn import manifold
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

label = {(0, 46):'pretrain', (47, 58):'Apple', (59, 65):'Bowl', (66, 74):'Bread', (75, 82):'Cup'}

def load_z(z_dir, file_except="None"):
    z_files = [os.path.join(z_dir, file) for file in sorted(os.listdir(z_dir)) if file.endswith(".json") and file.find(file_except)==-1]
    zs = [torch.load(zf, map_location='cpu')['z'].numpy() for zf in z_files]
    return zs, z_files

def tSNE(zs, dim=2):
    Zs = np.stack(zs, axis=0)
    tsne = TSNE(n_components=dim)
    manifold = tsne.fit_transform(Zs)
    return manifold.tolist()

def pca(zs, dim=2):
    Zs = np.stack(zs[75: 83], axis=0)
    _pca = PCA(n_components=dim)
    manifold = _pca.fit_transform(Zs)
    return manifold.tolist()

if __name__ == "__main__":
    zs, z_file = load_z("../Segmentation/checkpoints_objectpursuit_ithor_newtest_sequence/zs/")
    # manifold = tSNE(zs)
    manifold = pca(zs)
    print("result: ", manifold)