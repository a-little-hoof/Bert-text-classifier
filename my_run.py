import torch
from my_model import Model,Config
from my_utils import build_dataset, build_iterator
from my_train_eval import train, init_network
import numpy as np

if __name__ == '__main__':
    dataset = 'THUCNews'
    config = Config(dataset)

    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    print("Done!")

    bert_model = Model(config)
    bert_model.to(config.device)
    train(config, bert_model, train_iter, dev_iter, test_iter)