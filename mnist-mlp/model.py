from torch import nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(MLP,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class Config(object):
    def  __init__(self):
        self.train_batch_size = 64
        self.test_batch_size = 128
        self.lr = 0.01
        self.learning_rate = 0.01
        self.num_epoches = 20
        self.momentum = 0.5
        self.device = torch.device("cpu")