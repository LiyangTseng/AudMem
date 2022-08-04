import os
import csv
import torch
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import random
# set random seed to reproduce results
random.seed(0)
np.random.seed(0)

class Solver():
    ''' Solver for training'''
    def __init__(self, paras, **args):

        self.outdir = paras.outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.memo_output_path = os.path.join(self.outdir, "predicted_memorability_scores.csv")
        self.corr_output_path = os.path.join(self.outdir, "details.txt")


    def load_data(self):
        ''' Load data for testing '''
        self.labels_df = pd.read_csv("data/labels/track_memorability_scores_beta.csv")
        self.fold_num = 10
        fold_size = int(len(self.labels_df) / self.fold_num)
        self.train_set_list, self.test_set_list = [], []
        for fold_index in range(self.fold_num):
            testing_range = [ i for i in range(fold_index*fold_size, (fold_index+1)*fold_size)]
            for_test = self.labels_df.index.isin(testing_range)
            self.train_labels_df = self.labels_df[~for_test].reset_index(drop=True)
            self.test_labels_df = self.labels_df[for_test].reset_index(drop=True)
            self.train_set_list.append(self.train_labels_df)
            self.test_set_list.append(self.test_labels_df)


    def set_model(self):
        pass

    def exec(self):
        ''' Testing Memorabiliy Regression/Ranking System '''

        reg_loss_list = [], []
        for fold_idx in tqdm(range(self.fold_num)):
            self.pred_scores = [np.mean(self.train_set_list[fold_idx])] * len(self.test_set_list[fold_idx])
            self.labeled_scores = self.test_set_list[fold_idx].score.values

            reg_loss = torch.nn.MSELoss()(torch.tensor(self.pred_scores).unsqueeze(0), torch.tensor(self.labeled_scores).unsqueeze(0))
            reg_loss_list.append(reg_loss.item())


        with open(self.corr_output_path, 'w') as f:
            f.write("MSE loss: {}\n".format(str(np.mean(reg_loss_list))))

        print("regression loss: {}, saved at {}".format(np.mean(reg_loss_list), self.corr_output_path))
