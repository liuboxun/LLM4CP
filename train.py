import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
import scipy.io as sio
from models.GPT4CP import Model
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter
from metrics import NMSELoss, SE_Loss

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0001
epochs = 500
batch_size = 1024
device = torch.device('cuda')

best_loss = 100
save_path = "Weights/U2U_LLM4CP.pth"
train_TDD_r_path = "./H_U_his_train.mat"
train_TDD_t_path = "./H_U_pre_train.mat"
key = ['H_U_his_train', 'H_U_pre_train', 'H_D_pre_train']
train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=1, is_U2D=0, is_few=0)  # creat data for training
validate_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=0, is_U2D=0)  # creat data for validation

model = Model(gpu_id=0,
              pred_len=4, prev_len=16,
              UQh=1, UQv=1, BQh=1, BQv=1).to(device)
if os.path.exists(save_path):
    model = torch.load(save_path, map_location=device)


def save_best_checkpoint(model):  # save model function
    model_out_path = save_path
    torch.save(model, model_out_path)


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################
def train(training_data_loader, validate_data_loader):
    global epochs, best_loss
    print('Start training...')
    for epoch in range(epochs):
        epoch_train_loss, epoch_val_loss = [], []
        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            pred_t, prev = Variable(batch[0]).to(device), \
                           Variable(batch[1]).to(device)
            optimizer.zero_grad()  # fixed
            pred_m = model(prev, None, None, None)
            loss = criterion(pred_m, pred_t)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()
            optimizer.step()

        #       lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        print('Epoch: {}/{} training loss: {:.7f}'.format(epoch+1, epochs, t_loss))  # print loss for each epoch

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                pred_t, prev = Variable(batch[0]).to(device), \
                               Variable(batch[1]).to(device)
                optimizer.zero_grad()  # fixed
                pred_m = model(prev, None, None, None)
                loss = criterion(pred_m, pred_t)  # compute loss
                epoch_val_loss.append(loss.item())  # save all losses into a vector for one epoch
            v_loss = np.nanmean(np.array(epoch_val_loss))
            print('validate loss: {:.7f}'.format(v_loss))
            if v_loss < best_loss:
                best_loss = v_loss
                save_best_checkpoint(model)


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
    criterion = NMSELoss().to(device)
    train(training_data_loader, validate_data_loader)  # call train function (

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
