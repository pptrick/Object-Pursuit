import os
import numpy as np
from sklearn.manifold import TSNE

def load(base_dir, target_file, base_except="None"):
    base_files = [os.path.join(base_dir, file) for file in sorted(os.listdir(base_dir)) if file.endswith(".npy") and file.find(base_except)==-1]
    print(f"Load bases from {base_dir}: \n"
          f"{base_files}\n"
          f"{len(base_files)} bases in total")
    bases = [np.load(file) for file in base_files]
    target = np.load(target_file)
    return bases, target
    
def get_regress_coeff(bases, target):
    A = np.stack(bases, axis=1)
    coeff_mat = np.matmul(A.T, A)
    proj = np.matmul(A.T, target)
    coeff = np.matmul(np.linalg.inv(coeff_mat), proj)
    res = np.matmul(A, coeff)
    return coeff, res

def distance_r(src, tar):
    delta = tar - src
    return np.linalg.norm(delta)/np.linalg.norm(tar)

def manifold(bases, dim=2):
    Bases = np.stack(bases, axis=0)
    tsne = TSNE(n_components=dim)
    res = tsne.fit_transform(Bases)
    print(res)
    
    
if __name__ == "__main__":
    target_obj = 'Knife'
    print("target object: ", target_obj)
    bases, target = load("./Vec", f"./Vec/{target_obj}.npy", target_obj)
    print(f"target {target_obj} norm: ", np.linalg.norm(target))
    coeff, res = get_regress_coeff(bases, target)
    print(coeff)
    error = distance_r(res, target)
    print(error)
    # manifold(bases)
