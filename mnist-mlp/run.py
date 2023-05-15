import torch
from utils import build_dataset
from model import MLP,Config
from train_eval import train

config = Config()

train_loader,test_loader = build_dataset(config.train_batch_size,config.test_batch_size)
print(1)
model = MLP(28*28,300,100,10)
model.to(config.device)
train(config,model,train_loader,test_loader)