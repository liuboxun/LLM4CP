import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from einops import rearrange
from numpy import random


def noise(H, SNR):
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path_r, file_path_t, is_train=1, ir=1, SNR=15, is_U2D=0, is_few=0,
                 train_per=0.9, valid_per=0.1):
        super(Dataset_Pro, self).__init__()
        self.SNR = SNR
        self.ir = ir
        H_his = hdf5storage.loadmat(file_path_r)['H_U_his_train']  # v,b,l,k,a,b,c
        if is_U2D:
            H_pre = hdf5storage.loadmat(file_path_t)["H_D_pre_train"]  # v,b,l,k,a,b,c
        else:
            H_pre = hdf5storage.loadmat(file_path_t)["H_U_pre_train"]  # v,b,l,k,a,b,c
        # print(H_his.shape, H_pre.shape)

        batch = H_pre.shape[1]
        if is_train:
            H_his = H_his[:, :int(train_per * batch), ...]
            H_pre = H_pre[:, :int(train_per * batch), ...]
        else:
            H_his = H_his[:, int(train_per * batch):int((train_per + valid_per) * batch), ...]
            H_pre = H_pre[:, int(train_per * batch):int((train_per + valid_per) * batch), ...]
        H_his = rearrange(H_his, 'v n L k a b c -> (v n) L (k a b c)')
        H_pre = rearrange(H_pre, 'v n L k a b c -> (v n) L (k a b c)')

        B, prev_len, mul = H_his.shape
        _, pred_len, mul = H_pre.shape
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.seq_len = pred_len + prev_len

        dt_all = np.concatenate((H_his, H_pre), axis=1)
        np.random.shuffle(dt_all)
        H_his = dt_all[:, :prev_len, ...]
        H_pre = dt_all[:, -pred_len:, ...]
        for i in range(B):
            H_his[i, ...] = noise(H_his[i, ...], random.rand() * 15 + 5.0)
            H_pre[i, ...] = noise(H_pre[i, ...], random.rand() * 15 + 5.0)
        std = np.sqrt(np.std(np.abs(H_his) ** 2))
        H_his = H_his / std
        H_pre = H_pre / std
        H_pre = LoadBatch_ofdm(H_pre)
        H_his = LoadBatch_ofdm(H_his)
        if is_few == 1:
            H_pre = H_pre[::10, ...]
            H_his = H_his[::10, ...]
        self.pred = H_pre  # b,16,(48*2)
        self.prev = H_his  # b,4,(48*2)

    def __getitem__(self, index):
        return self.pred[index, :].float(), \
               self.prev[index, :].float()

    def __len__(self):
        return self.pred.shape[0]


def LoadBatch_ofdm_2(H):
    # H: B,T,K,mul     [tensor complex]
    # out:B,T,K,mul*2  [tensor real]
    B, T, K, mul = H.shape
    H_real = np.zeros([B, T, K, mul, 2])
    H_real[:, :, :, :, 0] = H.real
    H_real[:, :, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, K, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm_1(H):
    # H: B,T,mul     [tensor complex]
    # out:B,T,mul*2  [tensor real]
    B, T, mul = H.shape
    H_real = np.zeros([B, T, mul, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm(H, num=32):
    # H: B,T,mul             [tensor complex]
    # out:B*num,T,mul*2/num  [tensor real]
    B, T, mul = H.shape
    H = rearrange(H, 'b t (k a) ->(b a) t k', a=num)
    H_real = np.zeros([B * num, T, mul // num, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B * num, T, mul // num * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def Transform_TDD_FDD(H, Nt=4, Nr=4):
    # H: B,T,mul    [tensor real]
    # out:B',Nt,Nr  [tensor complex]
    H = H.reshape(-1, Nt, Nr, 2)
    H_real = H[..., 0]
    H_imag = H[..., 1]
    out = torch.complex(H_real, H_imag)
    return out
