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

mode = 1
cross_val = True
get_aic = True

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

            if mode == 2:
                train_f = train_f.reshape(-1, 15)
                train_label = train_label.reshape(1, -1)
                train_label = np.repeat(train_label, 100)
                val_f = val_f.reshape(-1, 15)

            model = LogisticRegression(max_iter=200, solver="newton-cg", penalty='l2')
            model.fit(train_f, train_label)

            if mode == 1:
                acc = model.score(val_f, val_label)
            elif mode == 2:
                pred = model.predict(val_f)
                pred = pred.reshape(-1, 100)

                acc = 0
                for i in range(pred.shape[0]):
                    acc += (val_label[i] == np.argmax(np.bincount(pred[i])))
                acc /= len(val_label)

                # pred = model.predict_proba(val_f)
                # pred = pred.reshape(-1, 20, 100).mean(-1) # (133, 20)
                # pred = np.argmax(pred, -1) # (133)
                # print(val_label.shape, pred.shape)
                # acc = (val_label == pred).sum() / len(val_label)
                # print(acc)
                # assert 0


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
        print(test_f.shape, test_label.shape)

        model = LogisticRegression(max_iter=200, solver="newton-cg", penalty="l2")
        model.fit(all_f, all_label)

        if get_aic:
            pred_train = model.predict_log_proba(all_f)
            print(pred_train.shape, all_label.shape)
            log_lik = []
            for j in range(pred_train.shape[0]):
                log_lik.append(pred_train[j][all_label[j]])
            log_lik = np.array(log_lik)
            # log_lik = np.sort(log_lik)[int(len(log_lik)/2):]
            print(np.sort(log_lik), log_lik.sum())
            # pred_train = model.predict(all_f).reshape(-1, 100)
            # print(pred_train.shape)
            # for j in range(pred_train.shape[0]):
            #     print(pred_train[j])
            #     print(np.argmax(np.bincount(pred_train[j])))
            #     print(all_label[j//100] == np.argmax(np.bincount(pred_train[j])))
            #     assert 0

        else:
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
            pd.DataFrame(result).to_csv("test_result_logistic_%d.csv"%mode, index=False)
