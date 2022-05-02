import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.solver import BaseSolver
from sklearn.svm import SVR
from src.dataset import PairHandCraftedDataset, HandCraftedDataset
from utils.calculate_handcrafted_features_stats import get_features_stats

CKPT_STEP = 10000

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.log_freq = self.config["experiment"]['log_freq']
        self.best_valid_loss = float('inf')

    def load_data(self):
        ''' Load data for training/validation '''
        
        self.labels_df = pd.read_csv(self.config["path"]["label_file"])
        # indexing except testing indices
        fold_size = int(len(self.labels_df) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        for_test = self.labels_df.index.isin(testing_range)
        self.labels_df = self.labels_df[~for_test]
        stats_dict = get_features_stats(label_df=self.labels_df, 
                                        features_dir=self.config["path"]["features_dir"], 
                                        for_test=False,
                                        features_dict = self.config["features"])

        self.train_labels_df = self.labels_df.sample(frac=1, random_state=self.paras.seed).reset_index(drop=True)
        self.train_set = HandCraftedDataset(labels_df=self.train_labels_df, config=self.config, stats_dict=stats_dict , pooling=True, split="train")


        """no augmentation"""
        # self.labels = self.train_set.scores
        # self.features = []
        # for feature_option in self.train_set.features_options:
        #     # TODO: add data augmentation
        #     seq_feat, non_seq_feat = feature_option[0]
        #     feat = torch.cat((seq_feat, non_seq_feat), dim=0)

        #     self.features.append(feat.numpy())
        
        """augmentation"""
        self.labels = []    
        self.features = []
        for i  in tqdm(range(len(self.train_set.features_options))):
            feature_option = self.train_set.features_options[i]
            for feats in feature_option:
                # concate seqential and unseqential data
                seq_feat, non_seq_feat = feats
                if non_seq_feat is not None:
                    feat = torch.cat((seq_feat, non_seq_feat), dim=0)
                else:
                    feat = seq_feat

                self.features.append(feat.numpy())
                self.labels.append(self.train_set.scores[i])
        print("len(self.features): ", len(self.features))

        
    def set_model(self):
        ''' Setup h_mlp model and optimizer '''
        # Model
        self.model = SVR(kernel=self.config["model"]["kernel"])
    
    def save_checkpoint(self):
        ckpt_path = os.path.join(self.ckpdir, self.paras.model+".pkl")
        with open(ckpt_path, 'wb') as fid:
            pickle.dump(self.model, fid)
        self.verbose("Saved checkpoint to {}".format(ckpt_path))

    def exec(self):
        # normalization already done in dataset
        self.model.fit(self.features, self.labels)
        self.save_checkpoint()

        