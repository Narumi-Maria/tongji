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
from sklearn.linear_model import LogisticRegression

cross_val = False
pelnaty = "l2"
solver = "newton-cg"

if __name__ == "__main__":

    ### loss
    criterion = nn.CrossEntropyLoss()

    img_list = np.array(glob.glob("./train/*"))
    train_label_dict = pd.read_csv("./label_train.csv", index_col=0).to_dict()["category"]
    K_cross = pickle.load(open("K_cross.pkl", "rb"))
    K = len(K_cross)

    all_f = []
    all_label = []
    for idx in range(len(img_list)):
        all_f.append(np.load(img_list[idx]).flatten().tolist())
        all_label.append(train_label_dict[img_list[idx].split("/")[-1]])
    all_f = np.array(all_f)
    all_label = np.array(all_label)

    if cross_val:
        bst_err_avg, bst_epoch_avg = [], []
        for k_val in range(K):

            ### data
            idx_val = K_cross[k_val]
            idx_train = np.array([i for i in range(len(img_list)) if i not in idx_val])
            assert len(np.intersect1d(idx_val, idx_train)) == 0

            train_f, train_label = all_f[idx_train], all_label[idx_train]
            val_f, val_label = all_f[idx_val], all_label[idx_val]

            model = LogisticRegression(max_iter=200, solver=solver, penalty=pelnaty)
            model.fit(train_f, train_label)

            acc = model.score(val_f, val_label)
            bst_err_avg.append(acc)
            print(k_val, acc)
            
        bst_err_avg = np.array(bst_err_avg)
        print(np.average(bst_err_avg), bst_err_avg)

    else:

        test_f = []
        test_label = []
        img_list_test = np.array(glob.glob("./test/*"))
        for idx in range(len(img_list_test)):
            test_label.append(img_list_test[idx].split("/")[-1])
            test_f.append(np.load(img_list_test[idx]).flatten().tolist())
        test_f = np.array(test_f)
        test_label = np.array(test_label)
        print(test_f.shape, test_label.shape)

        from sklearn.feature_selection import VarianceThreshold
        selector  = VarianceThreshold(threshold=5.)
        all_f = selector.fit_transform(all_f)
        print(all_f.shape)

        model = LogisticRegression(max_iter=200, solver=solver, penalty=pelnaty)
        model.fit(all_f, all_label)

        pred_train = model.predict_log_proba(all_f)
        print(pred_train.shape, all_label.shape)
        log_lik = []
        for j in range(pred_train.shape[0]):
            log_lik.append(pred_train[j][all_label[j]])
        log_lik = np.array(log_lik)
        print(np.sort(log_lik), log_lik.sum())
        print(model.n_features_in_)

        test_f = selector.transform(test_f)
        cls = model.predict(test_f)
        result = {"id": test_label, "category": cls}
        pd.DataFrame(result).to_csv("test_result_logistic_aic_var=5.csv", index=False)

       