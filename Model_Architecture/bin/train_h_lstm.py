import os
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.solver import BaseSolver
from src.optim import Optimizer
from models.memorability_model import H_LSTM
from src.dataset import PairHandCraftedDataset, HandCraftedDataset, Tabular_and_Sequential_Dataset
from src.util import human_format, get_grad_norm
from utils.calculate_handcrafted_features_stats import get_features_stats
from torch.utils.data import DataLoader

CKPT_STEP = 10000

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.log_freq = self.config["experiment"]['log_freq']
        self.use_ranking_loss = self.config["model"]["use_ranking_loss"]
        self.use_lds = self.config["model"]["use_lds"]
        self.use_fds = self.config["model"]["use_fds"]
        self.ranking_weight = config["model"]["ranking_weight"]
        self.best_valid_loss = float('inf')

    def fetch_data(self, data):
        ''' Move data to device '''

        if self.use_ranking_loss:
            feat_1, feat_2, lab_scores_1, lab_scores_2 = data

            seq_feat_1, non_seq_feat_1 = feat_1
            seq_feat_2, non_seq_feat_2 = feat_2

            seq_feat_1, non_seq_feat_1 = seq_feat_1.to(self.device), non_seq_feat_1.to(self.device)
            seq_feat_2, non_seq_feat_2 = seq_feat_2.to(self.device), non_seq_feat_2.to(self.device)

            lab_scores_1, lab_scores_2 = lab_scores_1.to(self.device).float(), lab_scores_2.to(self.device).float()
            return seq_feat_1, non_seq_feat_1, lab_scores_1, seq_feat_2, non_seq_feat_2, lab_scores_2
        else:
            seq_feats, non_seq_feats, lab_scores, weights = data
            # seq_feat, non_seq_feat = feat
            seq_feats, non_seq_feats = seq_feats.to(self.device).float(), non_seq_feats.to(self.device).float()
            lab_scores = lab_scores.to(self.device).float()
            weights = weights.to(self.device).float()
            return seq_feats, non_seq_feats, lab_scores, weights

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

        self.non_test_df = self.data_df[self.data_df['YT_id'].isin(train_yt_ids)].reset_index(drop=True)
        segment_nums = 9
        self.valid_df = self.non_test_df[:fold_size*segment_nums].reset_index(drop=True)
        self.train_df = self.non_test_df[fold_size*segment_nums:].reset_index(drop=True)

        self.train_set = Tabular_and_Sequential_Dataset(df=self.train_df, config=self.config, use_lds=self.use_lds)
        self.valid_set = Tabular_and_Sequential_Dataset(df=self.valid_df, config=self.config, use_lds=self.use_lds)
        


        self.write_log('train_distri/lab', self.train_df["label"].unique())
        self.write_log('valid_distri/lab', self.valid_df["label"].unique())
        
        self.verbose("generating weights for labels")
        # plot loss weight vs score
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(self.train_set.labels, self.train_set.weights_distri)
        plt.xlabel('score')
        plt.ylabel('loss weight')
        plt.title('train loss weight vs score')
        self.log.add_figure('loss_weight_vs_score/train', fig)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(self.valid_set.labels, self.valid_set.weights_distri)
        plt.xlabel('score')
        plt.ylabel('loss weight')
        plt.title('valid loss weight vs score')
        # add this figure to tensorboard
        self.log.add_figure('loss_weight_vs_score/valid', fig)
        plt.close()

        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.config["experiment"]["batch_size"],
                            num_workers=self.config["experiment"]["num_workers"], shuffle=True)
        self.valid_loader = DataLoader(dataset=self.valid_set, batch_size=self.config["experiment"]["batch_size"],
                            num_workers=self.config["experiment"]["num_workers"], shuffle=False)
        
        data_msg = ('I/O spec. | sequential feature dim = {}\t| non sequential feature dim = {}\t| use LDS: {}\t| use pitch_shift_augmentation: {}\t'
                .format(self.train_set[0][0][0].shape, self.train_set[0][1].shape, self.use_lds, self.paras.use_pitch_shift))

        self.verbose(data_msg)

    def set_model(self):
        ''' Setup h_lstm model and optimizer '''
        # Model
        self.model = H_LSTM(model_config=self.config["model"]).to(self.device)
        self.verbose(self.model.create_msg())

        # Losses
        self.reg_loss_func = nn.MSELoss(reduction="none") # regression loss
        self.rank_loss_func = nn.BCELoss() # ranking loss
        
        assert self.reg_loss_func.reduction == "none", "reduction need to be none for weighted loss"        

        # Optimizer
        # self.optimizer = Optimizer(self.model.parameters(), **self.config['hparas'])
        # self.verbose(self.optimizer.create_msg())

        self.optimizer = getattr(torch.optim, self.config["hparas"]["optimizer"]["type"])(self.model.parameters(), lr=self.config["hparas"]["optimizer"]["lr"])
        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt(cont=False)

    def backward(self, loss):
        '''
        Standard backward step with self.timer and debugger
        Arguments
            loss - the loss to perform loss.backward()
        '''
        self.timer.set()
        loss.backward()

        grad_norm = get_grad_norm(self.model.parameters())

        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step '+str(self.step))
        else:
            self.optimizer.step()
        self.timer.cnt('bw')
        return grad_norm

    def load_ckpt(self, cont=True):
        ''' Load ckpt if --load option is specified '''
        if self.paras.load:
            # Load weights
            ckpt = torch.load(
                self.paras.load, map_location=self.device if self.mode == 'train' else 'cpu')
            
            self.asr.load_state_dict(ckpt['model'])
            if self.emb_decoder is not None:
                self.emb_decoder.load_state_dict(ckpt['emb_decoder'])

            # Load task-dependent items
            if self.mode == 'train':
                if cont:
                    self.tts.load_state_dict(ckpt['tts'])
                    self.step = ckpt['global_step']
                    self.optimizer.load_opt_state_dict(ckpt['optimizer'])
                    self.verbose('Load ckpt from {}, restarting at step {}'.format(
                        self.paras.load, self.step))
            else:
                raise NotImplementedError

    def save_checkpoint(self, f_name, metric, score):
        ''''
        Ckpt saver
            f_name - <str> the name phnof ckpt file (w/o prefix) to store, overwrite if existed
            score  - <float> The value of metric used to evaluate model
        '''
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "model": self.model.state_dict(),
            # "optimizer": self.optimizer.get_opt_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.step,
            "global_epoch": self.epoch,
            metric: float(score)
        }

        torch.save(full_dict, ckpt_path)
        self.verbose("Saved checkpoint (epochs = {}, {} = {:.2f}) and status @ {}".
                    #  format(human_format(self.step), metric, score, ckpt_path))
                     format(human_format(self.epoch), metric, score, ckpt_path))

    def exec(self):
        ''' Training Memorabiliy Regression/Ranking System '''
        self.verbose('Total training steps {}.'.format(
            human_format(self.max_step)))
        self.timer.set()

        for self.epoch in range(self.max_epoch):
            print("\nepoch: {}/{}".format(self.epoch+1, self.max_epoch))
            
            self.model.train()
            train_reg_loss, train_rank_loss, train_total_loss = [], [], [] # record the loss of every batch
            train_reg_prediction, train_rank_prediction = [], []
            
            for i, data in enumerate(tqdm(self.train_loader)):
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                # tf_rate = self.optimizer.pre_step(self.step)
                # self.optimizer.opt.zero_grad()
                self.optimizer.zero_grad()
                total_loss = 0
                if self.use_ranking_loss:
                    # Fetch data
                    seq_feat_1, non_seq_feat_1, lab_scores_1, seq_feat_2, non_seq_feat_2, lab_scores_2 = self.fetch_data(data)
                    self.timer.cnt('rd')
                    lab_scores = torch.cat((lab_scores_1, lab_scores_2))


                    # Forward model
                    pred_scores_1 = self.model(seq_feat_1, non_seq_feat_1)
                    pred_scores_2 = self.model(seq_feat_2, non_seq_feat_2)
                    pred_scores = torch.cat((pred_scores_1, pred_scores_2))

                    reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    train_reg_prediction.append(pred_scores)
                    train_reg_loss.append(reg_loss.cpu().detach().numpy())
                    
                    # ref: https://www.cnblogs.com/little-horse/p/10468311.html
                    train_rank_prediction.append(pred_scores_1 > pred_scores_2)
                    pred_binary_rank = nn.Sigmoid()(pred_scores_1 - pred_scores_2)
                    lab_binary_rank = (pred_scores_1>pred_scores_2).float().to(self.device)
                    rank_loss = self.rank_loss_func(pred_binary_rank, lab_binary_rank)
                    train_rank_loss.append(rank_loss.cpu().detach().numpy())

                    total_loss = reg_loss + self.ranking_weight*rank_loss
                    train_total_loss.append(total_loss.cpu().detach().numpy())
                    self.timer.cnt('fw')

                    # Backprop
                    grad_norm = self.backward(total_loss)
                    self.step += 1

                    if i % self.log_freq == 0:
                        self.log.add_scalars('train_loss', {'reg_loss/train': np.mean(train_reg_loss)}, self.step)
                        self.log.add_scalars('train_loss', {'rank_loss/train': np.mean(train_rank_loss)}, self.step)
                        self.log.add_scalars('train_loss', {'total_loss/train': np.mean(train_total_loss)}, self.step)
    
                else:
                    seq_feats, non_seq_feats, lab_scores, weights = self.fetch_data(data)
                    self.timer.cnt('rd')

                    # Forward model
                    pred_scores = self.model(seq_feats, non_seq_feats)
                    total_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    total_loss *= weights.unsqueeze(1).expand_as(total_loss)
                    total_loss = torch.mean(total_loss)
                    
                    train_reg_prediction.append(pred_scores)
                    train_total_loss.append(total_loss.cpu().detach().numpy())
                    self.timer.cnt('fw')

                    # Backprop
                    grad_norm = self.backward(total_loss)
                    self.step += 1

            # Logger
            self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
                            .format(total_loss.cpu().item(), grad_norm, self.timer.show()))
            self.write_log('train_distri/pred', torch.cat(train_reg_prediction))
            self.write_log('MSE_loss', {'total_loss/train': np.mean(train_total_loss)})

            # Validation
            epoch_valid_total_loss = self.validate()
            if self.paras.patience:
                self.early_stopping(epoch_valid_total_loss, self.model)
                if self.early_stopping.early_stop:
                    self.verbose("Early Stoppoing")
                    break

            # End of step
            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
            torch.cuda.empty_cache()
            self.timer.set()
            # if self.step > self.max_step:
            #     break
        self.log.close()

    def validate(self):
        # Eval mode
        self.model.eval()
        valid_reg_loss, valid_rank_loss, valid_total_loss = [], [], [] # record the loss of every batch
        valid_reg_prediction, valid_rank_prediction = [], []

        for i, data in enumerate(self.valid_loader):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.valid_loader)))
            if self.use_ranking_loss:
                # Fetch data
                seq_feat_1, non_seq_feat_1, lab_scores_1, seq_feat_2, non_seq_feat_2, lab_scores_2 = self.fetch_data(data)
                lab_scores = torch.cat((lab_scores_1, lab_scores_2))

                # Forward model
                with torch.no_grad():
                    pred_scores_1 = self.model(seq_feat_1, non_seq_feat_1)
                    pred_scores_2 = self.model(seq_feat_2, non_seq_feat_2)
                    pred_scores = torch.cat((pred_scores_1, pred_scores_2))

                    valid_reg_prediction.append(pred_scores)
                    reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    valid_reg_loss.append(reg_loss.cpu().detach().numpy())

                    valid_rank_prediction.append(pred_scores_1 > pred_scores_2)
                    pred_binary_rank = nn.Sigmoid()(pred_scores_1 - pred_scores_2)
                    lab_binary_rank = (pred_scores_1>pred_scores_2).float().to(self.device)
                    rank_loss = self.rank_loss_func(pred_binary_rank, lab_binary_rank)
                    valid_rank_loss.append(rank_loss.cpu().detach().numpy())

                    total_loss = reg_loss + self.ranking_weight*rank_loss
                    valid_total_loss.append(total_loss.cpu().detach().numpy())

            else:
                seq_feats, non_seq_feats, lab_scores, weights = self.fetch_data(data)
                with torch.no_grad():
                    pred_scores = self.model(seq_feats, non_seq_feats)
                    total_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    total_loss *= weights.unsqueeze(1).expand_as(total_loss)
                    total_loss = torch.mean(total_loss)
                    
                    valid_reg_prediction.append(pred_scores)
                    valid_total_loss.append(total_loss.cpu().detach().numpy())

        if self.use_ranking_loss:
            epoch_valid_total_loss = np.mean(valid_total_loss)
            epoch_valid_reg_loss = np.mean(valid_reg_loss)
            epoch_valid_rank_loss = np.mean(valid_rank_loss)
            self.write_log('valid_distri/pred', torch.cat(valid_reg_prediction))
            self.write_log('loss', {'reg_loss/valid': epoch_valid_reg_loss})
            self.write_log('loss', {'rank_loss/valid': epoch_valid_rank_loss})
            self.write_log('loss', {'total_loss/valid': epoch_valid_total_loss})
        else:
            epoch_valid_total_loss = np.mean(valid_total_loss)
            self.write_log('valid_distri/pred', torch.cat(valid_reg_prediction))
            self.write_log('MSE_loss', {'total_loss/valid': epoch_valid_total_loss})
        # Ckpt if performance improves
        
        if epoch_valid_total_loss < self.best_valid_loss:
            self.best_valid_loss = epoch_valid_total_loss
            self.save_checkpoint('{}_best.pth'.format(
                self.paras.model), 'total_loss', epoch_valid_total_loss)



        # Resume training
        self.model.train()
        return epoch_valid_total_loss
