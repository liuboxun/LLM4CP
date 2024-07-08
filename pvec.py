import numpy as np
from numpy.linalg import pinv
from numpy.linalg import det
from numpy.linalg import svd
from numpy.linalg import norm
import matplotlib.pyplot as plt 
import matplotlib

def pronyvec(y, p = 4, pre_len = 3, startidx = 10, subcarriernum = 256, Nt = 2, Nr = 4):
    y = y.reshape([y.shape[0], y.shape[1], Nr, Nt])
    calH = np.zeros([subcarriernum*Nt*Nr, p], dtype=np.complex128)
    pL = np.zeros([subcarriernum*Nt*Nr, 1], dtype=np.complex128)
    for idx1 in range(p):
        for idx2 in range(Nt):
            for idx3 in range(subcarriernum):
                calH[idx2*Nr*subcarriernum + idx3*Nr: idx2*Nr*subcarriernum + (idx3+1)*Nr, idx1] = y[idx3, startidx-1-p+idx1, :, idx2]
    for idx2 in range(Nt):
        for idx3 in range(subcarriernum):
            pL[idx2*Nr*subcarriernum + idx3*Nr: idx2*Nr*subcarriernum + (idx3+1)*Nr, :] = np.expand_dims(y[idx3, startidx-1, :, idx2], axis=1)
    calH = np.matrix(calH)
    phat = -pinv(calH)*pL
    calH = np.hstack((calH[:, 1:p], pL))
    hpredict = -calH*phat
    hp1 = np.zeros([subcarriernum, Nr, Nt], dtype=np.complex128) 
    for idx1 in range(Nt):
        for idx2 in range(subcarriernum):
            hp1[idx2, :, idx1] = np.squeeze(hpredict[idx1*Nr*subcarriernum + idx2*Nr: idx1*Nr*subcarriernum + (idx2+1)*Nr, :])
    hp2 = np.zeros([subcarriernum, pre_len, Nr, Nt], dtype=np.complex128)
    hp2[:, 0, :, :] = hp1
    for idx1 in range(pre_len - 1):
        calH = np.hstack((calH[:, 1:p], hpredict))
        hpredict = -calH*phat
        for idx2 in range(Nt):
            for idx3 in range(subcarriernum):
                hp2[idx3, idx1+1, :, idx2] = np.squeeze(hpredict[idx2*Nr*subcarriernum + idx3*Nr: idx2*Nr*subcarriernum + (idx3+1)*Nr, :])
    hp2 = hp2.reshape([subcarriernum, pre_len, Nt*Nr])
    return hp2

