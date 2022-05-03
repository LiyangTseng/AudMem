import os
import csv
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from scipy import stats
from src.solver import BaseSolver
from src.dataset import HandCraftedDataset
from utils.calculate_handcrafted_features_stats import get_features_stats

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)

        self.outdir = paras.outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.memo_output_path = os.path.join(self.outdir, "predicted_memorability_scores.csv")
        self.corr_output_path = os.path.join(self.outdir, "details.txt")


    def load_data(self):
        ''' Load data for testing '''

        self.data_df = pd.read_csv(self.config["path"]["data_file"])

        self.data_df = self.data_df[self.data_df["augment_type"] == "original"]


        # find unique YT_ids
        YT_ids = self.data_df['YT_id'].unique()
        fold_size = int(len(YT_ids) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        test_yt_ids = [YT_ids[idx] for idx in testing_range]

        self.test_df = self.data_df[self.data_df['YT_id'].isin(test_yt_ids)]
        self.test_df = self.test_df.reset_index(drop=True)

        self.test_labels = self.test_df.iloc[:, -1].values
        self.test_features = self.test_df.iloc[:, 3:-1].values
        # noramlize features
        test_features_mean = np.mean(self.test_features, axis=0)
        test_features_std = np.std(self.test_features, axis=0)
        self.test_features = (self.test_features - test_features_mean) / test_features_std
        

    def set_model(self):
        with open(self.paras.load , 'rb') as fid:
            self.model = pickle.load(fid)  
        self.verbose("weight loaded from {}".format(self.paras.load))

    def exec(self):
        ''' Testing Memorabiliy Regression/Ranking System '''

        pred_scores = self.model.predict(self.test_features)

        with open(self.memo_output_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["YT_id", "segment_idx", "augment_type", "pred_score", "lab_score"])
            for idx in range(len(self.test_df)):
                writer.writerow([self.test_df.YT_id.values[idx],
                                self.test_df.segment_idx.values[idx],
                                self.test_df.augment_type.values[idx],
                                pred_scores[idx],
                                 self.test_labels[idx]])
        
            self.verbose("predicted memorability score saved at {}".format(self.memo_output_path))
        
        prediction_df = pd.read_csv(self.memo_output_path)
            
        # average prediction score for same YT_id and different segment_idx
        # prediction_df = prediction_df.groupby(["YT_id"]).mean()
        prediction_df = prediction_df.groupby(["YT_id"]).max()

        correlation = stats.spearmanr(prediction_df.pred_score.values, prediction_df.lab_score.values)
        reg_loss = torch.nn.MSELoss()(torch.tensor(prediction_df.pred_score.values).unsqueeze(0), torch.tensor(prediction_df.lab_score.values).unsqueeze(0))

        with open(self.corr_output_path, 'w') as f:
            f.write(str(correlation) + "\n")
            f.write("MSE loss: {}\n".format(str(reg_loss)))

        self.verbose("correlation result: {}, regression loss: {}, saved at {}".format(correlation, reg_loss, self.corr_output_path))