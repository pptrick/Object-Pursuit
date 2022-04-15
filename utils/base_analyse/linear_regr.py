import os
import numpy as np
   
def get_regress_coeff(bases, target):
    '''linear regression using least square'''
    A = np.stack(bases, axis=1)
    coeff_mat = np.matmul(A.T, A)
    proj = np.matmul(A.T, target)
    coeff = np.matmul(np.linalg.inv(coeff_mat), proj)
    res = np.matmul(A, coeff)
    return coeff, res

def distance_r(src, tar):
    delta = tar - src
    return np.linalg.norm(delta)/np.linalg.norm(tar)
