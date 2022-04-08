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
        self.labels_df = pd.read_csv(self.config["path"]["label_file"])
        
        fold_size = int(len(self.labels_df) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        for_test = self.labels_df.index.isin(testing_range)
        self.test_labels_df = self.labels_df[for_test].reset_index(drop=True)
        stats_dict = get_features_stats(label_df=self.test_labels_df, 
                                        features_dir=self.config["path"]["features_dir"], 
                                        for_test=True,
                                        features_dict = self.config["features"])

        self.test_set = HandCraftedDataset(labels_df=self.test_labels_df, config=self.config, stats_dict=stats_dict, pooling=True, split="test")
        
        self.features = []
        # concate seqential and unseqential data
        for feature_option in self.test_set.features_options:
            seq_feat, non_seq_feat = feature_option[0]
            feat = torch.cat((seq_feat, non_seq_feat), dim=0)

            self.features.append(feat.numpy())

        self.labels = self.test_set.scores

    def set_model(self):
        with open(self.paras.load , 'rb') as fid:
            self.model = pickle.load(fid)  
        self.verbose("weight loaded from {}".format(self.paras.load))

    def exec(self):
        ''' Testing Memorabiliy Regression/Ranking System '''

        pred_scores = self.model.predict(self.features)

        with open(self.memo_output_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["track", "pred_score", "lab_score"])
            for idx in range(len(self.labels)):
                # pred_score = self.model(feat).cpu().detach().item()
                # pred_score = 0.6560147067
                writer.writerow([self.test_labels_df.track.values[idx], pred_scores[idx], self.labels[idx]])
        
            self.verbose("predicted memorability score saved at {}".format(self.memo_output_path))
        
        prediction_df = pd.read_csv(self.memo_output_path)
        correlation = stats.spearmanr(prediction_df.pred_score.values, self.test_labels_df.score.values)
        reg_loss = torch.nn.MSELoss()(torch.tensor(prediction_df.pred_score.values).unsqueeze(0), torch.tensor(self.test_labels_df.score.values).unsqueeze(0))

        with open(self.corr_output_path, 'w') as f:
            f.write(str(correlation) + "\n")
            f.write("MSE loss: {}\n".format(str(reg_loss)))

        self.verbose("correlation result: {}, regression loss: {}, saved at {}".format(correlation, reg_loss, self.corr_output_path))