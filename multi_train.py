 #!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/10/29 17:37

import numpy as np
import random

import torch
import torch.nn as nn
from models import hetro_loss, mean_absolute_percentage_error, compute_rmse

# set random seed
SEED = 20202020

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('weak supervision task device', device)
eps=1e-6

def hetro_loss(x, mu, v):
    return (((x - mu) ** 2 / v) + torch.log(v)).mean()

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def compute_rmse(y_true, y_pred):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    return np.sqrt(((y_true/60 - y_pred/60) ** 2).mean())

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(model,optimizer,epoch,batch_size, train_data):

    train_path, train_ratio, train_slots, train_orderid, train_label =train_data
    road_idx=torch.tensor(list(range(num_locations))).to(device)
    time_idx=torch.tensor(list(range(num_times))).to(device)       
    model.train()
    tte_loss,speed_loss=0.0,0.0,0.0,0.0,0.0
    for i in range(0, train_path.shape[0],batch_size):
        paths_temp,ratio_temp,time_temp=train_path[i:i+batch_size],train_ratio[i:i+batch_size],train_slots[i:i+batch_size]
        optimizer.zero_grad()
        entire_out,mu,sigma=model(road_idx,time_idx,time_temp,g2,u,lengths,paths_temp,edge_type)
            # add 1
        
        mu=torch.cat([mu,torch.zeros(1,mu.shape[1]).to(device)])
        sigma=torch.cat([sigma,torch.zeros(1,sigma.shape[1]).to(device)])
        t=train_slots[i:i+batch_size]
        
        path_mu,path_sigma=mu[paths_temp],sigma[paths_temp]
        path_mu_new,path_mu_path_sigma=[],[]
        for o,l in enumerate(t):
            path_mu_new.append(path_mu[o:o+1,:,l])
            path_mu_path_sigma.append(path_sigma[o:o+1,:,l])
        path_mu,path_sigma=torch.cat(path_mu_new),torch.cat(path_mu_path_sigma)
    
        E_mu=torch.mul(path_mu,ratio_temp).sum(dim=1)+eps
        E_sigma=torch.mul(path_sigma,ratio_temp).pow(2).sum(dim=1)+eps

    
        label_time=train_label[i:i+batch_size,0:1]
        label_speed=train_label[i:i+batch_size,1]
        loss1=hetro_loss(label_speed,E_mu,E_sigma)
        loss2 = (torch.abs(entire_out - label_time) / label).mean()

        tte_loss+=torch.abs(entire_out-label_time).sum()
        speed_loss+=torch.abs(label_speed-E_mu).sum()
        
        print('\r tte loss: %f speed loss %f' %(tte_loss.item(), speed_loss), end="")

        (loss1+loss2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    print()
    print("Epoch:",epoch, tte_loss.item(),speed_loss.item())
    return tte_loss,speed_loss

def evaluate_traveltime(model, epoch,batch_size, test_data):
    model.eval()
    tte_loss,total_output,test_samples=0.0,0.0,[],0.0
    test_path,test_ratio,test_slots,test_orderid,test_label=test_data
    with torch.no_grad():
        for i in range(0, test_path.shape[0],batch_size):
            
            paths_temp,ratios_temp,time_temp=test_path[i:i+batch_size],test_ratio[i:i+batch_size],test_slots[i:i+batch_size]
            test_samples+=paths_temp.shape[0]
            entire_out,mu,sigma=model(road_idx,time_idx,time_temp,g2,u,lengths,paths_temp,edge_type)

            label_time=test_label[i:i+batch_size,0:1]
            total_output.append(entire_out)
            tte_loss+=torch.abs(entire_out-label_time).sum().item()
        tte_loss/=test_samples
        total_output=torch.cat(total_output)
        index=torch.unique(test_orderid)
        traj_output,label=torch.zeros(len(index),1),torch.zeros(len(index),1)
        for o, idx in enumerate(index):
            #max_lens=max(len(idx.reshape(-1)),max_lens)
            traj_output[o],label[o]=total_output[test_orderid==idx].sum(),test_label[test_orderid==idx].sum()
        traj_tte_loss=torch.abs(traj_output-label).mean()
       
        mse = compute_rmse(label, traj_output)
        mape = mean_absolute_percentage_error(label, traj_output)
        print("\n Epoch: %d, Path tte loss: %.4f, traj_tte_loss: %.4f" %( epoch,tte_loss/60,traj_tte_loss/60))
        return tte_loss,traj_tte_loss,mse,mape
