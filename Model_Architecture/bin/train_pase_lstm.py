import os
import json
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from src.solver import BaseSolver
from src.dataset import PairMemoWavDataset, MemoWavDataset
from models.pase_model import wf_builder
from models.memorability_model import LSTM
from tqdm import tqdm

from src.util import human_format, get_grad_norm
from torch.utils.data import DataLoader

CKPT_STEP = 10000
SAMPLING_RATE = 16000

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.use_ranking_loss = self.config["model"]["use_ranking_loss"]
        self.ranking_weight = config["model"]["ranking_weight"]


        with open(self.config["path"]["fe_cfg"], 'r') as fe_cfg_f:
            self.fe_cfg = json.load(fe_cfg_f)

        self.CUDA = True if self.paras.gpu else False
        self.best_valid_loss = float('inf')
        self.batch_size = self.config["experiment"]["batch_size"]
        
        self.encoder_ckpt = torch.load(self.config["path"]["encoder_ckpt"], map_location=self.device)
        self.encoder_weights = self.encoder_ckpt["model"]
        self.encoder_mode = self.config["experiment"]["encoder_mode"]

    def load_data(self):
        ''' Load data for training/validation '''
      
        # get labels from csv file
        self.labels_df = pd.read_csv(self.config["path"]["label_file"])
        # indexing except testing indices
        fold_size = int(len(self.labels_df) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        for_test = self.labels_df.index.isin(testing_range)
        self.labels_df = self.labels_df[~for_test]
        self.labels_df = self.labels_df.sample(frac=1, random_state=self.paras.seed).reset_index(drop=True)
        self.valid_labels_df = self.labels_df[:fold_size].reset_index(drop=True)
        self.train_labels_df = self.labels_df[fold_size:].reset_index(drop=True)

        if self.use_ranking_loss:
            self.train_set = PairMemoWavDataset(self.train_labels_df,
                                        self.config["path"]["data_root"][0],
                                        self.config["path"]["data_cfg"][0], 
                                        'train',
                                        sr=SAMPLING_RATE,
                                        preload_wav=self.config["dataset"]["preload_wav"],
                                        same_sr=True)

            self.valid_set = PairMemoWavDataset(self.valid_labels_df,
                                        self.config["path"]["data_root"][0],
                                        self.config["path"]["data_cfg"][0], 
                                        'valid',
                                        sr=SAMPLING_RATE,
                                        preload_wav=self.config["dataset"]["preload_wav"],
                                        same_sr=True)

        else:
            self.train_set = MemoWavDataset(self.train_labels_df,
                                        self.config["path"]["data_root"][0],
                                        self.config["path"]["data_cfg"][0], 
                                        'train',
                                        sr=SAMPLING_RATE,
                                        preload_wav=self.config["dataset"]["preload_wav"],
                                        same_sr=True)

            self.valid_set = MemoWavDataset(self.valid_labels_df,
                                        self.config["path"]["data_root"][0],
                                        self.config["path"]["data_cfg"][0], 
                                        'valid',
                                        sr=SAMPLING_RATE,
                                        preload_wav=self.config["dataset"]["preload_wav"],
                                        same_sr=True)

        self.bpe = (self.train_set.total_wav_dur // self.config["dataset"]["chunk_size"]) // self.config["experiment"]["batch_size"]
        self.verbose("Train data length: {}".format(self.train_set.total_wav_dur/16000/3600.0))
        self.verbose("Valid data length: {}".format(self.valid_set.total_wav_dur/16000/3600.0))

        self.train_loader = DataLoader(self.train_set, batch_size=self.config["experiment"]["batch_size"],
                                        shuffle=True,
                                        num_workers=self.config["experiment"]["num_workers"],drop_last=True,
                                        pin_memory=self.CUDA)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.config["experiment"]["batch_size"],
                                        shuffle=True,
                                        num_workers=self.config["experiment"]["num_workers"],drop_last=True,
                                        pin_memory=self.CUDA)

    def fetch_data(self, data):
        ''' Move data to device '''
        if self.use_ranking_loss:
            wavs_1, wavs_2, scores_1, scores_2 = data
            
            wavs_1 = wavs_1.view(wavs_1.size(0), 1, -1).to(self.device)
            wavs_2 = wavs_2.view(wavs_2.size(0), 1, -1).to(self.device)
            scores_1 = scores_1.to(self.device).float()
            scores_2 = scores_2.to(self.device).float()
            
            return wavs_1, wavs_2, scores_1, scores_2
        else:
            wav, score = data
            wav = wav.to(self.device)
            score = score.to(self.device).float()
            return wav, score



    def set_model(self):
        ''' Setup downstream memorability model and optimizer '''

        self.log_freq = self.config["experiment"]['log_freq']
        # Losses
        self.reg_loss_func = nn.MSELoss()
        self.rank_loss_func = nn.BCELoss() # ranking loss

        # Downstream Model
        options = self.config["model"]
        print(options)        
        inp_dim = options["input_dim"]

        self.verbose("using pase_lstm")
        self.downstream_model = LSTM(options,inp_dim).to(self.device)
        

        # Pre-train Encoder
        self.verbose("loading pase encoder...")
        self.encoder = wf_builder(self.fe_cfg).to(self.device)
        if self.encoder_mode == "frozen":
            self.encoder.load_state_dict(self.encoder_weights)
            self.encoder.eval() 
    
            self.optimizer = getattr(torch.optim, self.config["hparas"]["optimizer"]["type"])(
                                                                self.downstream_model.parameters(), 
                                                                lr=self.config["hparas"]["optimizer"]["lr"])

        elif self.encoder_mode == "fine-tune":
            self.encoder.load_state_dict(self.encoder_weights)
            self.encoder.train()
            # set optimizer to encoder & downstream model
            self.optimizer = getattr(torch.optim, self.config["hparas"]["optimizer"]["type"])(
                                                                list(self.encoder.parameters())+list(self.downstream_model.parameters()),
                                                                lr=self.config["hparas"]["optimizer"]["lr"])

        elif self.encoder_mode == "from-scratch":
            self.encoder.train()

            self.optimizer = getattr(torch.optim, self.config["hparas"]["optimizer"]["type"])(
                                                                list(self.encoder.parameters())+list(self.downstream_model.parameters()), 
                                                                lr=self.config["hparas"]["optimizer"]["lr"])
        else:
            raise Exception("Not Implement Error")

    def backward(self, loss):
        '''
        Standard backward step with self.timer and debugger
        Arguments
            loss - the loss to perform loss.backward()
        '''
        self.timer.set()
        loss.backward()

        grad_norm = get_grad_norm(self.downstream_model.parameters())

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
            "encoder": self.encoder.state_dict(),
            "model": self.downstream_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # "global_step": self.step,
            "global_epoch": self.epoch,
            metric: float(score)
        }

        torch.save(full_dict, ckpt_path)
        self.verbose("Saved checkpoint (step = {}, {} = {:.2f}) and status @ {}".
                    #  format(human_format(self.step), metric, score, ckpt_path))
                     format(human_format(self.epoch), metric, score, ckpt_path))

    def exec(self):
        ''' Downstream model training '''

        self.timer.set()

        for self.epoch in range(self.max_epoch):
            print("\nepoch: {}/{}".format(self.epoch+1, self.max_epoch))
            
            self.encoder.train()
            self.downstream_model.train()
            train_reg_loss, train_rank_loss, train_total_loss = [], [], [] 
            train_reg_prediction, train_rank_prediction = [], []

            for i, data in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()

                if self.use_ranking_loss:

                    wavs_1, wavs_2, lab_scores_1, lab_scores_2 = self.fetch_data(data)
                    lab_scores = torch.cat((lab_scores_1, lab_scores_2))
                    self.timer.cnt('rd')

                    # inference
                    features_1 = self.encoder(wavs_1, self.device)
                    pred_scores_1, _ = self.downstream_model(features_1)

                    features_2 = self.encoder(wavs_2, self.device)
                    pred_scores_2, _ = self.downstream_model(features_2)

                    pred_scores = torch.cat((pred_scores_1, pred_scores_2))

                    train_reg_prediction.append(pred_scores)
                    
                    reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    train_reg_loss.append(reg_loss.cpu().detach().numpy())

                    # ref: https://www.cnblogs.com/little-horse/p/10468311.html
                    train_rank_prediction.append(pred_scores_1 > pred_scores_2)
                    pred_binary_rank = nn.Sigmoid()(pred_scores_1 - pred_scores_2)
                    lab_binary_rank = torch.unsqueeze((lab_scores_1>lab_scores_2), 1).float().to(self.device)
                    rank_loss = self.rank_loss_func(pred_binary_rank, lab_binary_rank)
                    train_rank_loss.append(rank_loss.cpu().detach().numpy())

                    total_loss = reg_loss + self.ranking_weight*rank_loss
                    train_total_loss.append(total_loss.cpu().detach().numpy())
                    
                    self.timer.cnt('fw')
                    grad_norm = self.backward(total_loss)
                    self.step += 1

                    if i % self.log_freq == 0:
                        self.log.add_scalars('train_loss', {'reg_loss/train': np.mean(train_reg_loss)}, self.step)
                        self.log.add_scalars('train_loss', {'rank_loss/train': np.mean(train_rank_loss)}, self.step)
                        self.log.add_scalars('train_loss', {'total_loss/train': np.mean(train_total_loss)}, self.step)
                else:
                    wavs, lab_scores = self.fetch_data(data)
                    self.timer.cnt('rd')

                    # inference
                    features = self.encoder(wavs, self.device)
                    pred_scores, _ = self.downstream_model(features)

                    train_reg_prediction.append(pred_scores)
                    
                    reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    train_reg_loss.append(reg_loss.cpu().detach().numpy())

                    self.timer.cnt('fw')
                    grad_norm = self.backward(reg_loss)
                    self.step += 1

                    if i % self.log_freq == 0:
                        self.log.add_scalars('train_loss', {'reg_loss/train': np.mean(train_reg_loss)}, self.step)
                        self.log.add_scalars('train_loss', {'total_loss/train': np.mean(train_total_loss)}, self.step)

            

            self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
                            .format(total_loss.cpu().item(), grad_norm, self.timer.show()))
            # Logger
            self.write_log('train_distri/pred', torch.cat(train_reg_prediction))

            # Validation
            epoch_valid_total_loss = self.validate()
            if self.paras.patience:
                self.early_stopping(epoch_valid_total_loss, self.downstream_model)
                if self.early_stopping.early_stop:
                    self.verbose("Early Stoppoing")
                    break

            # End of step
            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
            torch.cuda.empty_cache()
            self.timer.set()

        self.log.close()


    def validate(self):
        # evaluation
        self.encoder.eval()
        self.downstream_model.eval()
        
        if self.use_ranking_loss:
            with torch.no_grad():
                valid_reg_loss, valid_rank_loss, valid_total_loss = [], [], [] # record the loss of every batch
                valid_reg_prediction, valid_rank_prediction = [], []

                for i, data in enumerate(self.valid_loader):
                    self.progress('Valid step - {}/{}'.format(i+1, len(self.valid_loader)))
                    # Fetch data
                    wavs_1, wavs_2, lab_scores_1, lab_scores_2 = self.fetch_data(data)
                    lab_scores = torch.cat((lab_scores_1, lab_scores_2))

                    # inference
                    features_1 = self.encoder(wavs_1, self.device)
                    pred_scores_1, _ = self.downstream_model(features_1)

                    features_2 = self.encoder(wavs_2, self.device)
                    pred_scores_2, _ = self.downstream_model(features_2)

                    pred_scores = torch.cat((pred_scores_1, pred_scores_2))

                    valid_reg_prediction.append(pred_scores)
                    
                    reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    valid_reg_loss.append(reg_loss.cpu().detach().numpy())

                    # ref: https://www.cnblogs.com/little-horse/p/10468311.html
                    valid_rank_prediction.append(pred_scores_1 > pred_scores_2)
                    pred_binary_rank = nn.Sigmoid()(pred_scores_1 - pred_scores_2)
                    lab_binary_rank = torch.unsqueeze((lab_scores_1>lab_scores_2), 1).float().to(self.device)
                    rank_loss = self.rank_loss_func(pred_binary_rank, lab_binary_rank)
                    valid_rank_loss.append(rank_loss.cpu().detach().numpy())

                    total_loss = reg_loss + self.ranking_weight*rank_loss
                    valid_total_loss.append(total_loss.cpu().detach().numpy())
        
        else:
            with torch.no_grad():
                valid_reg_loss, valid_total_loss = [], []
                valid_reg_prediction = []

                for i, data in enumerate(self.valid_loader):
                    self.progress('Valid step - {}/{}'.format(i+1, len(self.valid_loader)))
                    # Fetch data
                    wavs, lab_scores = self.fetch_data(data)

                    # inference
                    features = self.encoder(wavs, self.device)
                    pred_scores, _ = self.downstream_model(features)

                    valid_reg_prediction.append(pred_scores)
                    
                    reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    valid_reg_loss.append(reg_loss.cpu().detach().numpy())

                    total_loss = reg_loss
                    valid_total_loss.append(total_loss.cpu().detach().numpy())

        if self.use_ranking_loss:
            epoch_valid_total_loss = np.mean(valid_total_loss)
            epoch_valid_reg_loss = np.mean(valid_reg_loss)
            epoch_valid_rank_loss = np.mean(valid_rank_loss)

            self.write_log('valid_distri/pred', torch.cat(valid_reg_prediction))
            self.write_log('valid_loss', {'reg_loss/valid': epoch_valid_reg_loss})
            self.write_log('valid_loss', {'rank_loss/valid': epoch_valid_rank_loss})
            self.write_log('valid_loss', {'total_loss/valid': epoch_valid_total_loss})

        else:
            epoch_valid_total_loss = np.mean(valid_total_loss)
            epoch_valid_reg_loss = np.mean(valid_reg_loss)

            self.write_log('valid_distri/pred', torch.cat(valid_reg_prediction))
            # self.write_log('valid_loss', {'reg_loss/valid': epoch_valid_reg_loss})
            self.write_log('valid_loss', {'total_loss/valid': epoch_valid_total_loss})

        # Ckpt if performance improves
        
        if epoch_valid_total_loss < self.best_valid_loss:
            self.best_valid_loss = epoch_valid_total_loss
            self.save_checkpoint('{}_best.pth'.format(
                self.paras.model), 'total_loss', epoch_valid_total_loss)

        # Regular ckpt
        self.save_checkpoint('epoch_{}.pth'.format(
            self.epoch), 'total_loss', epoch_valid_total_loss)


        # Resume training
        self.encoder.train()
        self.downstream_model.train()
        return epoch_valid_total_loss
