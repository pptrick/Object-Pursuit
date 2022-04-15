'''view latent code z in low-dimensional manifold'''
import os
from sklearn import manifold
import torch
import random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_z(z_path, file_except="None"):
    if os.path.isdir(z_path):
        z_files = [os.path.join(z_path, file) for file in sorted(os.listdir(z_path)) if file.endswith(".json") and file.find(file_except)==-1]
        zs = [torch.load(zf, map_location='cpu')['z'].numpy() for zf in z_files]
        return zs, z_files
    elif os.path.isfile(z_path):
        if z_path.endswith(".pth"): # checkpoint for multinet
            zs = torch.load(z_path, map_location='cpu')['z']
            return zs, z_path
    else:
        return None, None

def tSNE(zs, dim=2):
    Zs = np.stack(zs, axis=0)
    tsne = TSNE(n_components=dim)
    manifold = tsne.fit_transform(Zs)
    return manifold.tolist()

def pca(zs, dim=2):
    zs = np.stack(zs, axis=0)
    _pca = PCA(n_components=dim)
    manifold = _pca.fit_transform(zs)
    return manifold.tolist()

def shuffle(zs):
    index = [i for i in range(len(zs))]
    random.shuffle(index)
    return zs[index]