import numpy as np
from numpy.core.shape_base import vstack, hstack
from numpy.linalg import pinv, svd, norm
from numpy import expand_dims, floor, abs
import matplotlib.pyplot as plt 
import matplotlib
import math
import scipy.io as scio
import pvec

pi = math.pi

def DFT(N):
    row = np.arange(N).reshape(1, N)
    column = np.arange(N).reshape(N, 1)
    W = 1 / np.sqrt(N) * np.exp(-2j * pi / N * np.dot(column, row))
    return W

def PAD(y, p = 10, pre_len = 3, startidx = 10, subcarriernum = 256, Nt = 2, Nr = 4):
    y = y.reshape([y.shape[0], y.shape[1], Nr, Nt])
    L = int(floor(p/2) * 2 - 1)
    N = int((L + 1) / 2)
    S = np.kron(DFT(subcarriernum), DFT(Nt))
    gamma = 0.1
    #hp1 = np.zeros([subcarriernum, Nr, Nt], dtype=np.complex128)
    hp2 = np.zeros([subcarriernum, pre_len, Nr, Nt], dtype=np.complex128)
    for Nridx in range(Nr):
        ypad = np.zeros([subcarriernum * Nt, y.shape[1]], dtype=np.complex128)
        for idx in range(subcarriernum):
            ypad[idx*Nt:(idx+1)*Nt, :] = np.squeeze(y[idx, :, Nridx, :]).T
        gu = S.T.conj().dot(ypad[:, startidx-p])
        guselect = abs(gu)
        gusort = np.sort(guselect)[::-1]
        guidx = np.argsort(guselect)[::-1]
        Ns = subcarriernum
        gusort = gusort[0:Ns]
        guidx = guidx[0:Ns]
        ghat = np.zeros([subcarriernum * Nt, pre_len], dtype=np.complex128)            
        g = S.T.conj().dot(ypad)
        for Nsidx in range(Ns):
            rn = guidx[Nsidx]
            calG = np.zeros([N, N], dtype=np.complex128)
            for idx in range(N):
                calG[idx, :] = g[rn, startidx-p+idx:startidx-p+idx+N]
            boldg = g[rn, startidx-p+N:startidx-p+2*N].T
            phat = np.dot(-pinv(calG), boldg)
            gnew = g[rn, startidx-p+L-N+1:startidx-p+L+1]
            for timeidx in range(pre_len):
                ghat[rn, timeidx] = np.dot(-gnew, phat)    
                gnew = hstack((gnew[1:N], ghat[rn, timeidx]))
                calG = hstack((calG[:, 1:N], vstack((np.expand_dims(calG[1:N, N-1], axis=1), ghat[rn, timeidx]))))
                phat = np.dot(-pinv(calG), gnew.T)
        hhat = S.dot(ghat)
        for idx in range(subcarriernum):
            hp2[idx, :, Nridx, :] = hhat[idx*Nt:(idx+1)*Nt, :].T
    hp2 = hp2.reshape([subcarriernum, pre_len, Nt*Nr])
    return hp2

def PAD2(y, p = 10, pre_len = 3, startidx = 10, subcarriernum = 256, Nt = 2, Nr = 4):
    y = y.reshape([y.shape[0], y.shape[1], Nr, Nt])
    L = int(floor(p/2) * 2 - 1)
    N = int((L + 1) / 2)
    S = np.kron(DFT(subcarriernum), DFT(Nt))
    gamma = 0.1

    hp2 = np.zeros([subcarriernum, pre_len, Nr, Nt], dtype=np.complex128)
    for Nridx in range(Nr):
        ypad = np.zeros([subcarriernum * Nt, y.shape[1]], dtype=np.complex128)
        for idx in range(subcarriernum):
            ypad[idx*Nt:(idx+1)*Nt, :] = np.squeeze(y[idx, :, Nridx, :]).T
        gu = S.T.conj().dot(ypad[:, startidx-p])
        guselect = abs(gu)
        gusort = np.sort(guselect)[::-1]
        guidx = np.argsort(guselect)[::-1]
        Ns = 128
        gusort = gusort[0:Ns]
        guidx = guidx[0:Ns]
        ghat = np.zeros([subcarriernum * Nt, pre_len], dtype=np.complex128)            
        g = S.T.conj().dot(ypad)
        for Nsidx in range(Ns):
            rn = guidx[Nsidx]
            calG = np.zeros([N, N], dtype=np.complex128)
            for idx in range(N):
                calG[idx, :] = g[rn, startidx-p+idx:startidx-p+idx+N]
            boldg = g[rn, startidx-p+N:startidx-p+2*N].T
            phat = np.dot(-pinv(calG), boldg)
            gnew = g[rn, startidx-p+L-N+1:startidx-p+L+1]
            for timeidx in range(pre_len):
                ghat[rn, timeidx] = np.dot(-gnew, phat)    
                gnew = hstack((gnew[1:N], ghat[rn, timeidx]))
        hhat = S.dot(ghat)
        for idx in range(subcarriernum):
            hp2[idx, :, Nridx, :] = hhat[idx*Nt:(idx+1)*Nt, :].T
    hp2 = hp2.reshape([subcarriernum, pre_len, Nt*Nr])
    return hp2

def PAD3(y, p = 10, pre_len = 3, startidx = 10, subcarriernum = 256, Nt = 2, Nr = 4):
    y = y.reshape([y.shape[0], y.shape[1], Nr, Nt])
    S = np.kron(DFT(subcarriernum), DFT(Nt))
    hp2 = np.zeros([subcarriernum, pre_len, Nr, Nt], dtype=np.complex128)
    for Nridx in range(Nr):
        ypad = np.zeros([subcarriernum * Nt, y.shape[1]], dtype=np.complex128)
        for idx in range(subcarriernum):
            ypad[idx*Nt:(idx+1)*Nt, :] = np.squeeze(y[idx, :, Nridx, :]).T
        gu = S.T.conj().dot(ypad)
        gu = gu.reshape([y.shape[0], y.shape[3], y.shape[1]]).transpose(0, 2, 1)
        gu = np.expand_dims(gu, axis=2)
        ghat = pvec.pronyvec(gu, p, pre_len, startidx, subcarriernum, Nt, 1)
        ghat = np.reshape(ghat.transpose(0, 2, 1), [y.shape[0]*y.shape[3], pre_len, 1]).squeeze()
        hhat = S.dot(ghat)
        for idx in range(subcarriernum):
            hp2[idx, :, Nridx, :] = hhat[idx*Nt:(idx+1)*Nt, :].T
    hp2 = hp2.reshape([subcarriernum, pre_len, Nt*Nr])
    return hp2
