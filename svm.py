import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

mode = 1
cross_val = True

if __name__ == "__main__":

    ### loss
    criterion = nn.CrossEntropyLoss()

    img_list = np.array(glob.glob("./train/*"))
    train_label_dict = pd.read_csv("./label_train.csv", index_col=0).to_dict()["category"]
    K_cross = pickle.load(open("K_cross.pkl", "rb"))
    K = len(K_cross)

    bst_err_avg, bst_epoch_avg = [], []

    all_f = []
    all_label = []
    for idx in range(len(img_list)):
        all_f.append(np.load(img_list[idx]).flatten().tolist())
        all_label.append(train_label_dict[img_list[idx].split("/")[-1]])
    all_f = np.array(all_f)
    all_label = np.array(all_label)

    # from sklearn import preprocessing
    # scaler = preprocessing.StandardScaler().fit(all_f)
    # all_f = scaler.transform(all_f)

    # from sklearn.feature_selection import VarianceThreshold
    # selector = VarianceThreshold(threshold=1.0)
    # all_f = selector.fit_transform(all_f)
    # print(all_f.shape, all_label.shape)

    if cross_val:

        for k_val in range(K):

            ### data
            idx_val = K_cross[k_val]
            idx_train = np.array([i for i in range(len(img_list)) if i not in idx_val])
            assert len(np.intersect1d(idx_val, idx_train)) == 0

            train_f, train_label = all_f[idx_train], all_label[idx_train]
            val_f, val_label = all_f[idx_val], all_label[idx_val]

            if mode == 2:
                train_f = train_f.reshape(-1, 15)
                train_label = train_label.reshape(1, -1)
                train_label = np.repeat(train_label, 100)
                val_f = val_f.reshape(-1, 15)

            clf = svm.SVC(decision_function_shape='ovo', kernel='sigmoid')
            clf.fit(train_f, train_label)
            clf.decision_function_shape = "ovr"

            if mode == 1:
                acc = clf.score(val_f, val_label)
            elif mode == 2:
                pred = clf.predict(val_f)
                pred = pred.reshape(-1, 100)
                acc = 0
                for i in range(pred.shape[0]):
                    acc += (val_label[i] == np.argmax(np.bincount(pred[i])))
                acc /= len(val_label)
            bst_err_avg.append(acc)
            print(k_val, acc)
            
        bst_err_avg = np.array(bst_err_avg)
        print(np.average(bst_err_avg), bst_err_avg)
    
    else:
        if mode == 2:
            all_f = all_f.reshape(-1, 15)
            all_label = all_label.reshape(1, -1)
            all_label = np.repeat(all_label, 100)
            print(all_f.shape, all_label.shape)

        test_f = []
        test_label = []
        img_list_test = np.array(glob.glob("./test/*"))
        for idx in range(len(img_list_test)):
            test_label.append(img_list_test[idx].split("/")[-1])
            test_f.append(np.load(img_list_test[idx]).flatten().tolist())
        test_f = np.array(test_f)
        if mode == 2:
            test_f = test_f.reshape(-1, 15)
        test_label = np.array(test_label)

        model = svm.SVC(decision_function_shape='ovo')
        model.fit(all_f, all_label)
        model.decision_function_shape = "ovr"

        if mode == 1:
            cls = model.predict(test_f)
        elif mode == 2:
            test_pred = model.predict(test_f)
            test_pred = test_pred.reshape(-1, 100)
            print(test_pred.shape)
            
            cls = []
            for i in range(test_pred.shape[0]):
                cls.append(np.argmax(np.bincount(test_pred[i])))

        result = {"id": test_label, "category": cls}
        pd.DataFrame(result).to_csv("test_result_svm_%d.csv"%mode, index=False)