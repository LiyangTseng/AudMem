import os
import csv
import torch
import numpy as np
import pandas as pd
from scipy import stats
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
        self.test_labels_df = pd.read_csv("data/labels/track_memorability_scores_beta.csv")
        
    def set_model(self):
        pass

    def exec(self):
        ''' Testing Memorabiliy Regression/Ranking System '''

        # test 1000 times to get the average result
        correlation_list, reg_loss_list = [], []
        for i in range(1000):
            self.pred_scores = [random.random() for _ in range(len(self.test_labels_df))]

            correlation = stats.spearmanr(self.pred_scores, self.test_labels_df.score.values)
            reg_loss = torch.nn.MSELoss()(torch.tensor(self.pred_scores).unsqueeze(0), torch.tensor(self.test_labels_df.score.values).unsqueeze(0))
            correlation_list.append(correlation[0])
            reg_loss_list.append(reg_loss)

        with open(self.corr_output_path, 'w') as f:
            f.write("spearman's correlation: {}\n".format(str(np.mean(correlation_list))))
            f.write("MSE loss: {}\n".format(str(np.mean(reg_loss_list))))

        print("correlation result: {}, regression loss: {}, saved at {}".format(correlation, reg_loss, self.corr_output_path))
