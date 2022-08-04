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
from src.util import _prepare_weights

CKPT_STEP = 10000

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.log_freq = self.config["experiment"]['log_freq']
        self.best_valid_loss = float('inf')

    def load_data(self):
        ''' Load data for training/validation '''
        
        self.data_df = pd.read_csv(self.config["path"]["data_file"])
        if not self.paras.use_pitch_shift:
            # only use original audio
            self.verbose("Only use original audio")
            self.data_df = self.data_df[self.data_df["augment_type"] == "original"]
        
        YT_ids = self.data_df['YT_id'].unique()
        fold_size = int(len(YT_ids) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        train_yt_ids = [YT_ids[idx] for idx in range(len(YT_ids)) if idx not in testing_range]

        self.train_df = self.data_df[self.data_df['YT_id'].isin(train_yt_ids)]
        self.train_df = self.train_df.reset_index(drop=True)
        
        self.train_labels = self.train_df.iloc[:, -1].values
        self.train_features = self.train_df.iloc[:, 3:-1].values
        # noramlize features
        train_features_mean = np.mean(self.train_features, axis=0)
        train_features_std = np.std(self.train_features, axis=0)
        self.train_features = (self.train_features - train_features_mean) / train_features_std
        self.use_lds = self.config["model"]["use_lds"]

        self.verbose("Loaded data for training/validation")

        if self.use_lds:
            self.weights_distri = _prepare_weights(labels=self.train_labels, 
                                                    reweight="inverse", 
                                                    max_target=1, 
                                                    lds=True,
                                                    lds_ks=self.config["hparas"]["lds_ks"], 
                                                    lds_sigma=self.config["hparas"]["lds_sigma"],
                                                    bin_size=self.config["hparas"]["bin_size"])
        data_msg = ('I/O spec.  | use LDS: {}\t| use pitch_shift_augmentation: {}\t'
                .format(self.use_lds, self.paras.use_pitch_shift))

        self.verbose(data_msg)

    
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
        if self.use_lds:
            self.model.fit(self.train_features, self.train_labels, sample_weight=self.weights_distri)
        else:
            self.model.fit(self.train_features, self.train_labels)
        self.save_checkpoint()

        