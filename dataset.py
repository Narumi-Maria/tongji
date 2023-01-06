import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, img_list, split="train"):

        self.split = split
        self.img_list = img_list
        self.train_label = pd.read_csv("./label_train.csv", index_col=0).to_dict()["category"]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        feature = np.load(self.img_list[idx]) # (100, 15)
        if self.split == "train":
            label = self.train_label[self.img_list[idx].split("/")[-1]]
        else:
            label = self.img_list[idx].split("/")[-1]

        return feature, label