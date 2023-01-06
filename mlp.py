import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import dataset
from models import *

NUM_EPOCH = 76

if __name__ == "__main__":

    ### data
    img_list = np.array(glob.glob("./train/*"))
    train_set = dataset(img_list=img_list)
    train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)

    ### loss
    criterion = nn.CrossEntropyLoss()

    ### model
    model = MLP()
    model = model.cuda()

    ### optimizer
    param_group = [{'params': model.parameters()}]
    optimizer = torch.optim.AdamW(
                    param_group, 
                    lr=1e-4,
                    weight_decay=1e-7,
                    eps=1e-8
                )

    ### train
    model.train()

    log_train_loss = []
    for epoch in tqdm(range(NUM_EPOCH + 1)):
        for i, batch in enumerate(train_loader):

            feature, label = batch
            feature = feature.float().cuda()
            label = label.long().cuda()

            out = model(feature)
            train_loss = criterion(out, label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            log_train_loss.append(train_loss.detach().cpu().numpy())

    log_train_loss = np.array(log_train_loss)
    print(np.average(log_train_loss), log_train_loss)

    ### eval
    model.eval()

    ### data
    img_list = np.array(glob.glob("./test/*"))
    test_set = dataset(img_list, "test")
    test_loader = DataLoader(test_set, batch_size=2048, shuffle=False)
    print("test set length:", len(test_set))

    result = {"id": [], "category": []}
    for i, batch in enumerate(test_loader):

        feature, ids = batch
        feature = feature.float().cuda()

        out = model(feature)
        _, cls = torch.max(out, -1)

        result["id"].extend(list(ids))
        result["category"].extend(cls.cpu().numpy())

    pd.DataFrame(result).to_csv("test_result_mlp.csv", index=False)


        
        
