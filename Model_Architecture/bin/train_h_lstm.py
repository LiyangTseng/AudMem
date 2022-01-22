import os
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.solver import BaseSolver
from src.optim import Optimizer
from models.memorability_model import H_LSTM
from src.dataset import HandCraftedDataset
from src.util import human_format, get_grad_norm
from torch.utils.data import DataLoader

CKPT_STEP = 10000

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.ranking_weight = config["model"]["ranking_weight"]
        self.best_valid_loss = float('inf')

    def fetch_data(self, data):
        ''' Move data to device '''
        seq_feat, non_seq_feat, labeled_scores = data
        seq_feat, non_seq_feat, labeled_scores = seq_feat.to(self.device), non_seq_feat.to(self.device), labeled_scores.to(self.device)

        return seq_feat, non_seq_feat, labeled_scores

    def load_data(self):
        ''' Load data for training/validation '''
        self.train_set = HandCraftedDataset(config=self.config, pooling=False, mode="train")
        self.valid_set = HandCraftedDataset(config=self.config, pooling=False, mode="valid")
        self.write_log('train_distri/lab', self.train_set.filename_memorability_df["score"].values)
        self.write_log('valid_distri/lab', self.valid_set.filename_memorability_df["score"].values)

        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.config["experiment"]["batch_size"],
                            num_workers=self.config["experiment"]["num_workers"], shuffle=True)
        self.valid_loader = DataLoader(dataset=self.valid_set, batch_size=self.config["experiment"]["batch_size"],
                            num_workers=self.config["experiment"]["num_workers"], shuffle=False)
        
        data_msg = ('I/O spec.  | audio feature = {}\t| sequential feature dim = {}\t| nonsequential feature dim = {}\t'
                .format(self.train_set.features_dict, self.train_set.sequential_features[0].shape, self.train_set.non_sequential_features[0].shape))

        self.verbose(data_msg)

    def set_model(self):
        ''' Setup h_lstm model and optimizer '''
        # Model
        self.model = H_LSTM(model_config=self.config["model"]).to(self.device)
        self.verbose(self.model.create_msg())

        dataiter = iter(self.train_loader)
        data = dataiter.next()
        seq_feat, non_seq_feat, _ = self.fetch_data(data)

        # don't know why need to wrap data in another list yet, but it works
        self.write_log(self.model, [seq_feat, non_seq_feat])

        # Losses
        self.reg_loss_func = nn.MSELoss() # regression loss
        self.rank_loss_func = nn.BCELoss() # ranking loss
        

        # Optimizer
        self.optimizer = Optimizer(self.model.parameters(), **self.config['hparas'])
        self.verbose(self.optimizer.create_msg())

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
            "optimizer": self.optimizer.get_opt_state_dict(),
            # "global_step": self.step,
            "global_epoch": self.epoch,
            metric: float(score)
        }

        torch.save(full_dict, ckpt_path)
        self.verbose("Saved checkpoint (step = {}, {} = {:.2f}) and status @ {}".
                    #  format(human_format(self.step), metric, score, ckpt_path))
                     format(human_format(self.epoch), metric, score, ckpt_path))

    def exec(self):
        ''' Training Memorabiliy Regression/Ranking System '''
        self.verbose('Total training steps {}.'.format(
            human_format(self.max_step)))
        self.timer.set()

        for epoch in range(self.max_epoch):
            print("\nepoch: {}/{}".format(epoch+1, self.max_epoch))
            
            self.model.train()
            train_reg_loss, train_rank_loss, train_total_loss = [], [], [] # record the loss of every batch
            train_reg_prediction, train_rank_prediction = [], []
            
            for i, data in enumerate(self.train_loader):
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                # tf_rate = self.optimizer.pre_step(self.step)
                self.optimizer.opt.zero_grad()
                total_loss = 0

                # Fetch data
                seq_feat, non_seq_feat, lab_scores = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward model
                pred_scores = self.model(seq_feat, non_seq_feat)
                reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                train_reg_prediction.append(pred_scores)
                train_reg_loss.append(reg_loss.cpu().detach().numpy())
                
                train_rank_prediction.append(pred_scores[:seq_feat.size(0)//2] > pred_scores[seq_feat.size(0)//2:])
                pred_binary_rank = nn.Sigmoid()(pred_scores[:seq_feat.size(0)//2] - pred_scores[seq_feat.size(0)//2:])
                lab_binary_rank = (pred_scores[:seq_feat.size(0)//2]>pred_scores[seq_feat.size(0)//2:]).float().to(self.device)
                rank_loss = self.rank_loss_func(pred_binary_rank, lab_binary_rank)
                train_rank_loss.append(rank_loss.cpu().detach().numpy())

                total_loss = reg_loss + self.ranking_weight*rank_loss
                train_total_loss.append(total_loss.cpu().detach().numpy())
                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(total_loss)
                self.step += 1

    
            epoch_train_total_loss = np.mean(train_total_loss)
            epoch_train_reg_loss = np.mean(train_reg_loss)
            epoch_train_rank_loss = np.mean(train_rank_loss)

            # Logger
            self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
                            .format(total_loss.cpu().item(), grad_norm, self.timer.show()))
            self.write_log('train_distri/pred', torch.cat(train_reg_prediction))

            self.write_log('loss', {'reg_loss/train': epoch_train_reg_loss})
            self.write_log('loss', {'rank_loss/train': epoch_train_rank_loss})
            self.write_log('loss', {'total_loss/train': epoch_train_total_loss})

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
            self.epoch += 1
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
            # Fetch data
            seq_feat, non_seq_feat, lab_scores = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                pred_scores = self.model(seq_feat, non_seq_feat)
                valid_reg_prediction.append(pred_scores)
                reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                valid_reg_loss.append(reg_loss.cpu().detach().numpy())

                valid_rank_prediction.append(pred_scores[:seq_feat.size(0)//2] > pred_scores[seq_feat.size(0)//2:])
                pred_binary_rank = nn.Sigmoid()(pred_scores[:seq_feat.size(0)//2] - pred_scores[seq_feat.size(0)//2:])
                lab_binary_rank = (pred_scores[:seq_feat.size(0)//2]>pred_scores[seq_feat.size(0)//2:]).float().to(self.device)
                rank_loss = self.rank_loss_func(pred_binary_rank, lab_binary_rank)
                valid_rank_loss.append(rank_loss.cpu().detach().numpy())

                total_loss = reg_loss + self.ranking_weight*rank_loss
                valid_total_loss.append(total_loss.cpu().detach().numpy())


        epoch_valid_total_loss = np.mean(valid_total_loss)
        epoch_valid_reg_loss = np.mean(valid_reg_loss)
        epoch_valid_rank_loss = np.mean(valid_rank_loss)
        self.write_log('valid_distri/pred', torch.cat(valid_reg_prediction))
        self.write_log('loss', {'reg_loss/valid': epoch_valid_reg_loss})
        self.write_log('loss', {'rank_loss/valid': epoch_valid_rank_loss})
        self.write_log('loss', {'total_loss/valid': epoch_valid_total_loss})
        
        # Ckpt if performance improves
        
        if epoch_valid_total_loss < self.best_valid_loss:
            self.best_valid_loss = epoch_valid_total_loss
            self.save_checkpoint('{}_{}.pth'.format(
                self.paras.model, self.epoch), '{}_loss'.format(self.paras.model), epoch_valid_total_loss)

        # Regular ckpt
        self.save_checkpoint('epoch_{}.pth'.format(
            self.epoch), 'h_lstm_loss', epoch_valid_total_loss)


        # Resume training
        self.model.train()
        return epoch_valid_total_loss
