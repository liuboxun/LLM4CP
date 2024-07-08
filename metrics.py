# =======================================================================================================================
# =======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import hdf5storage


# =======================================================================================================================
# =======================================================================================================================


class SE_Loss(nn.Module):
    def __init__(self, snr=10, device=torch.device("cuda:0")):
        super().__init__()
        self.SNR = snr
        self.device = device

    def forward(self, h, h0):
        # input : h:  B, Nt, Nr (complex)      h0: B, Nt, Nr (complex)
        # 1. prepare data
        SNR = self.SNR
        B, Nt, Nr = h.shape
        H = h.to(self.device)  # B * Nr * Nt
        H0 = h0.to(self.device)  # B * Nr * Nt
        if Nr != 1:
            S_real = torch.diag(torch.ones(Nr, 1).squeeze()).unsqueeze(0).repeat([B, 1, 1])  # b,2 * 2
        elif Nr == 1:
            S_real = torch.diag(torch.ones(Nr, 1)).unsqueeze(0).repeat([B, 1, 1])  # b,2 * 2
        S_imag = torch.zeros([B, Nr, Nr])
        S = torch.complex(S_real, S_imag).to(device=self.device)
        matmul0 = torch.matmul(H0, S)
        fro = torch.norm(matmul0, p='fro', dim=(1, 2))  # B,1
        noise_var = (torch.pow(fro, 2) / (Nt * Nr)) * pow(10, (-SNR / 10))
        # 2. get D and D0
        D = torch.adjoint(H)
        D = torch.div(D, torch.norm(D, p=2, dim=(1, 2), keepdim=True))
        D0 = torch.adjoint(H0)
        D0 = torch.div(D0, torch.norm(D0, p=2, dim=(1, 2), keepdim=True))
        # 3. get SE and SE0
        matmul1 = torch.matmul(D, H0)
        matmul2 = torch.matmul(D0, H0)

        noise_var = noise_var.unsqueeze(1).unsqueeze(1)  # B,1,1
        SE = -torch.log2(torch.det(torch.div(torch.pow(torch.abs(matmul1), 2), noise_var) + S))  # B
        SE = torch.mean(SE.real)

        SE0 = -torch.log2(torch.det(torch.div(torch.pow(torch.abs(matmul2), 2), noise_var) + S))  # B
        SE0 = torch.mean(SE0.real)

        return SE, SE0


def NMSE_cuda(x_hat, x):
    power = torch.sum(x ** 2)
    mse = torch.sum((x - x_hat) ** 2)
    nmse = mse / power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse
