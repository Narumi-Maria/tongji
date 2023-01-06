import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

### data, K-cross val
K = 10
K_cross = [[] for _ in range(K)]
img_list = np.array(glob.glob("./train/*"))
train_label = pd.read_csv("./label_train.csv", index_col=0).to_dict()["category"]
Label2Idx = {i:[] for i in range(20)}
for i, img in enumerate(img_list):
    label = train_label[img.split("/")[-1]]
    Label2Idx[label].append(i)
Label2Idx = {k:np.array(v) for k,v in Label2Idx.items()}
for label in range(20):
    np.random.shuffle(Label2Idx[label])
    split = np.linspace(0, len(Label2Idx[label]), K + 1, dtype=np.int32)
    k_shuffle = np.array(range(len(split)-1))
    np.random.shuffle(k_shuffle)
    for k in k_shuffle:
        K_cross[k_shuffle[k]].extend(Label2Idx[label][split[k]:split[k+1]])

for k, item in enumerate(K_cross):
    cnt = np.zeros(20)
    for id in item:
        cnt[train_label[img_list[id].split("/")[-1]]] += 1
    print(k, cnt)

pickle.dump(K_cross, open("K_cross.pkl", "wb"))