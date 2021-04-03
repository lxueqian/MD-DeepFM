#!/usr/bin/env python
import os
import math
import numpy as np
import csv    
import pandas as pd
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
continous_features = 12


# lives = pd.read_csv('data/live.txt').iloc[:, :].values
# lives = sorted(lives,key=lambda x:x[-1]) # Lives的相关信息
# live_nums = len(lives)
# print('所有live种类数：',live_nums)

# # 用户订阅的lives信息，输出字典，键为string类型的用户编号
# with open('data/raw/ordered_traindata.json','r') as f:
#     alldata = json.loads(f.read())
# print('所用用户数量：',len(alldata))

class CriteoDataset(Dataset):
    """
    Custom dataset class for Criteo dataset in order to use efficient 
    dataloader tool provided by PyTorch.
    """ 
    def __init__(self, root, train=True):
        """
        Initialize file path and train/test mode.

        Inputs:
        - root: Path where the processed data file stored.
        - train: Train or test. Required.
        """
        self.root = root
        self.train = train

        if not self._check_exists:
            raise RuntimeError('Dataset not found.')

        if self.train:
            data = pd.read_csv(self.root)
            self.train_data = data.iloc[:, :-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(self.root)
            self.test_data = data.iloc[:, :-1].values
    
    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            Xv = torch.from_numpy(np.ones_like(dataI))
            return Xi, Xv, targetI
        else:
            dataI = self.test_data.iloc[idx, :]
            Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            Xv = torch.from_numpy(np.ones_like(dataI))
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)
        
        
class DeepFM(nn.Module):
    """
    A DeepFM network with RMSE loss for rates prediction problem.

    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this 
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.

    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[32, 32], num_classes=2, dropout=[0.5, 0.5], 
                 use_cuda=True, verbose=False):
        """
        Initialize a new network

        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.float
        self.savepig = []

        """
            check if use cuda
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        """
            init fm part
        """

        # fm_first_order_Linears = nn.ModuleList(
        #         [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[:26]])
        # fm_first_order_embeddings = nn.ModuleList(
        #         [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[26:36]])
        # self.fm_first_order_models = fm_first_order_Linears.extend(fm_first_order_embeddings)


        fm_second_order_Linears = nn.ModuleList(
                [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[:26]])
        fm_second_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[26:36]])
        self.fm_second_order_models = fm_second_order_Linears.extend(fm_second_order_embeddings)

        """
            init deep part
        """
        all_dims = [self.field_size * self.embedding_size] + \
            self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_'+str(i),
                    nn.Linear(all_dims[i-1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_'+str(i),
                    nn.Dropout(dropout[i-1]))

    def forward(self, Xi, Xv):
        """
        Forward process of network. 

        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        """
            fm part
        """
#         emb = self.fm_first_order_models[20]
# #        print(Xi.size())
#         for num in Xi[:, 20, :][0]:
#             if num > self.feature_sizes[20]:
#                 print("index out")

#         fm_first_order_emb_arr = []
#         for i, emb in enumerate(self.fm_first_order_models):
#             if i <=25:
#                 Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.float)
#                 fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
#             else:
#                 Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.long)
#                 fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())
# #        print("successful")      
# #        print(len(fm_first_order_emb_arr))
#         fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation

        fm_second_order_emb_arr = []
        for i, emb in enumerate(self.fm_second_order_models):
            if i <= 25:
                # print(Xi[:, i, :])

                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                # print(emb(Xi_tem))
                # print(emb(Xi_tem).unsqueeze(1))
                # print(torch.sum(emb(Xi_tem).unsqueeze(1), 1))
                # print((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]))
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
                
            else:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                # print(i)
                # print(Xi[:, i, :])
                # print(emb(Xi_tem))
                # print((torch.sum(emb(Xi_tem), 1)))
                # print((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]))
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())
                # exit()
        
        fm_first_order = torch.cat(fm_second_order_emb_arr, 1)

        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
            fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [
            item*item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(
            fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square -
                           fm_second_order_emb_square_sum) * 0.5
        """
            deep part
        """
#        print(len(fm_second_order_emb_arr))
#        print(torch.cat(fm_second_order_emb_arr, 1).shape)
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
#            print("successful") 
        """
            sum
        """
#        print("1",torch.sum(fm_first_order, 1).shape)
#        print("2",torch.sum(fm_second_order, 1).shape)
#        print("deep",torch.sum(deep_out, 1).shape)
#        print("bias",bias.shape)
        bias = torch.nn.Parameter(torch.randn(Xi.size(0)))
        total_sum = torch.sum(fm_first_order, 1) + \
                    torch.sum(fm_second_order, 1) + \
                    torch.sum(deep_out, 1) + bias
        return total_sum

    def fit(self, loader_train, loader_val, optimizer, epochs=1, verbose=False, print_every=512):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations. 
        """
        """
            load input data
        """
        model = self.train().to(device=self.device)
        criterion = F.binary_cross_entropy_with_logits
        for epoch in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                # print(xi[0:])

                y = y.to(device=self.device, dtype=self.dtype)
                
                total = model(xi, xv)
#                print(total.shape)
#                print(y.shape)
                loss = criterion(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % print_every == 0:
                    print('Epoch %d Iteration %d, loss = %.4f' % (epoch, t, loss.item()))
                    print()

                    AUC = self.check_accuracy(loader_val, model)
                    self.savepig.append([epoch,AUC,loss.item()])
                    print()

        print(self.savepig)
        with open('performance.json','w') as f:
            f.write(json.dumps(self.savepig))

        # self.plot_line()


    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')   
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                xv = xv.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)
                total = model(xi, xv)
                preds = (F.sigmoid(total) > 0.5).to(dtype=self.dtype)
                pred = F.sigmoid(total)
#                print(preds.dtype)
#                print(y.dtype)
#                print(preds.eq(y).cpu().sum())
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
#                print("successful")
            acc = float(num_correct) / num_samples
            fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
            AUC = auc(fpr, tpr)
            
            print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
            print("AUC:",AUC)

            return AUC


    def plot_line(self):
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        x=[]
        y=[]
        for item in self.savepig:
            x.append(item[0])
            y.append(item[1])

        plt.plot(x, y, 'ro')
        # plt.show()
        plt.savefig('result.png')



def predict():
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load('model/deepfm.pkl'))

    model.eval()
    with torch.no_grad():
        for xi, xv in loader:
            xi = xi.to(device=self.device, dtype=self.dtype)  # move to device,e.g. GPU
            xv = xv.to(device=self.device, dtype=torch.float)
            total = model(xi, xv)
            preds = (torch.sigmoid(total) > 0.5)
            num_samples += preds.size(0)


def main():

    # 900000 items for training, 10000 items for valid, of all 1000000 items
    # Num_train = 676450
    Num_train = 432540

    # load data
    train_data = CriteoDataset('./data/train_0901.txt', train=True)
    loader_train = DataLoader(train_data, batch_size=1024,
                            sampler=sampler.SubsetRandomSampler(range(Num_train)))
    val_data = CriteoDataset('./data/test_0901.txt', train=True)
    loader_val = DataLoader(val_data, batch_size=1024,
                            sampler=sampler.SubsetRandomSampler(range(0, 43254)))

    # loader_val = DataLoader(val_data, batch_size=50)

    feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
    feature_sizes = [int(x) for x in feature_sizes]
    print(feature_sizes)

    model = DeepFM(feature_sizes, use_cuda=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.fit(loader_train, loader_val, optimizer, epochs=200, verbose=True)

    print('****** end ******')


if __name__ == "__main__":
    main()