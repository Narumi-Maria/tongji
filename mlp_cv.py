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

mode = 1

if __name__ == "__main__":


    K_cross = pickle.load(open("K_cross.pkl", "rb"))
    K = len(K_cross)
    img_list = np.array(glob.glob("./train/*"))

    ### loss
    criterion = nn.CrossEntropyLoss()

    bst_err_avg, bst_epoch_avg = [], []
    for k_val in range(K):

        ### data
        idx_val = K_cross[k_val]
        idx_train = np.array([i for i in range(len(img_list)) if i not in idx_val])
        assert len(np.intersect1d(idx_val, idx_train)) == 0
        train_set = dataset(img_list=img_list[idx_train])
        val_set = dataset(img_list=img_list[idx_val])

        train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=2048, shuffle=False)

        ### model
        if mode == 1:
            model = MLP()
        elif mode == 2:
            model = MLP_per_frame()
        elif mode == 3:
            model = RNN()
        model = model.cuda()

        ### optimizer
        param_group = [{'params': model.parameters()}]
        optimizer = torch.optim.AdamW(
                        param_group, 
                        lr=1e-4,
                        weight_decay=1e-7,
                        eps=1e-8
                    )

        bst_acc = 0.
        bst_epoch = 0

        for epoch in tqdm(range(100)):

            ### train
            model.train()
            for i, batch in enumerate(train_loader):

                feature, label = batch
                feature = feature.float().cuda()
                label = label.long().cuda()
                if mode == 2:
                    label = label.unsqueeze(1).expand(-1, 100).flatten(0, 1)

                out = model(feature)
                assert out.shape[-1] == 20
                if mode == 2:
                    out = out.flatten(0, 1)
        
                train_loss = criterion(out, label)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            
            ### val
            model.eval()
            val_acc = 0.
            for i, batch in enumerate(val_loader):

                feature, label = batch
                feature = feature.float().cuda()
                label = label.long().cuda()

                out = model(feature)
                assert out.shape[-1] == 20

                _, cls = torch.max(out, -1)

                if mode == 2:
                    cls = cls.cpu().numpy()
                    label = label.cpu().numpy()
                    for j in range(cls.shape[0]):
                        val_acc += (label[j] == np.argmax(np.bincount(cls[j])))
                else:
                    val_acc += (cls==label).sum().cpu().numpy()

            val_acc /= float(len(val_set))

            if val_acc > bst_acc:
                bst_acc = val_acc
                bst_epoch = epoch

        print(k_val, bst_epoch, bst_acc)

        bst_err_avg.append(bst_acc)
        bst_epoch_avg.append(bst_epoch)

    bst_err_avg = np.array(bst_err_avg)
    bst_epoch_avg = np.array(bst_epoch_avg)
    print(np.average(bst_err_avg), bst_err_avg)
    print(np.average(bst_epoch_avg), bst_epoch_avg)