import os
import math
import json
import torch
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.solver import BaseSolver
from src.optim import Optimizer
from models.ast_models import ASTModel
from src.dataset import AST_AudioDataset
from src.util import human_format, get_grad_norm
from torch.utils.data import DataLoader, WeightedRandomSampler

CKPT_STEP = 10000

class Solver(BaseSolver):
    ''' Solver for training'''

    def parse_yaml(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                self.parse_yaml(value)
            else:
                setattr(self, key, value)


    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        if not os.path.exists(self.config["path"]["pretrained_mdl_path"]):
            os.system("wget {} - O {}".format("https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1", self.config["path"]["pretrain_weight_file"]))
        

        self.parse_yaml(self.config)

        self.audio_conf = {'num_mel_bins': self.num_mel_bins, 'target_length': self.target_length, 'freqm': self.freqm, 'timem': self.timem, 'mixup': self.mixup, 'dataset': self.dataset,
                    'mode':'train', 'mean': self.dataset_mean, 'std': self.dataset_std, 'noise': self.noise}

        self.val_audio_conf = {'num_mel_bins': self.num_mel_bins, 'target_length': self.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': self.dataset,
                        'mode': 'evaluation', 'mean': self.dataset_mean, 'std': self.dataset_std, 'noise': False}



        self.log_freq = self.config["experiment"]['log_freq']
        self.use_ranking_loss = self.config["model"]["use_ranking_loss"]
        self.ranking_weight = config["model"]["ranking_weight"]
        self.best_valid_loss = float('inf')


    def fetch_data(self, data):
        ''' Move data to device '''
        # input in shape [batch_size, input_tdim, input_fdim]


        if self.use_ranking_loss:
            raise Exception ("Not implemented yet")
            img_1, img_2, lab_scores_1, lab_scores_2 = data

            img_1, img_2 = img_1.to(self.device), img_2.to(self.device)
            lab_scores_1, lab_scores_2 = lab_scores_1.to(self.device).float(), lab_scores_2.to(self.device).float()

            return img_1, img_2, lab_scores_1, lab_scores_2
        else:
            fbank, label = data
            fbank = fbank.to(self.device)
            label = label.to(self.device).float()
            return fbank, label

    def generate_json_format(self, df, split):
        """ format should be 'data': {'wav': '', 'labels': ''}, ... """
        assert split in ['train', 'valid', 'test']
        
        audio_root = os.path.abspath(self.config["path"]["audio_root"])
        if split != "test":
            audio_subdir = os.listdir(audio_root)
        else:
            audio_subdir = ["original"]
        data_arr = df.to_numpy()
        
        data = {}
        data["data"] = []
        
        for row in data_arr:
            wav, label = row
            for subdir in audio_subdir:
                data["data"].append({"wav": os.path.join(audio_root, subdir, wav), "labels": label})
        
        json_input_dir = self.config["path"]["ssast_input_dir"]
        os.makedirs(json_input_dir, exist_ok=True)        
        json_data_path = os.path.join(json_input_dir, split+'.json')
        
        with open(json_data_path, 'w') as f:
            json.dump(data, f)
        
        self.verbose("{} data saved at {}".format(split, json_data_path))


    def load_data(self):
        ''' Load data for training/validation '''
        self.labels_df = pd.read_csv(self.config["path"]["label_file"])
        # indexing except testing indices
        fold_size = int(len(self.labels_df) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        for_test = self.labels_df.index.isin(testing_range)
        self.test_labels_df = self.labels_df[for_test].reset_index(drop=True)
        
        self.labels_df = self.labels_df[~for_test] # extract non-testing indices
        # shuffling to make validation set
        self.labels_df = self.labels_df.sample(frac=1, random_state=self.paras.seed).reset_index(drop=True)
        self.valid_labels_df = self.labels_df[:fold_size].reset_index(drop=True)
        self.train_labels_df = self.labels_df[fold_size:].reset_index(drop=True)
        self.write_log('train_distri/lab', self.train_labels_df.score.values)
        self.write_log('valid_distri/lab', self.valid_labels_df.score.values)
        
        
        # generate json data for this fold
        self.generate_json_format(self.train_labels_df, "train")
        self.generate_json_format(self.valid_labels_df, "valid")
        self.generate_json_format(self.test_labels_df, "test")

        # if use balanced sampling, note - self-supervised pretraining should not use balance sampling as it implicitly leverages the label information.
        if self.bal == 'bal':
            print('balanced sampler is being used')
            samples_weight = np.loadtxt(self.data_train[:-5]+'_weight.csv', delimiter=',')
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

            self.train_loader = DataLoader(
                AST_AudioDataset(self.data_train, audio_conf=self.audio_conf, config=self.config),
                batch_size=self.config["experiment"]["batch_size"], sampler=sampler, num_workers=self.config["experiment"]["num_workers"], pin_memory=False, drop_last=True)
        else:
            print('balanced sampler is not used')
            self.train_loader = DataLoader(
                AST_AudioDataset(self.data_train, audio_conf=self.audio_conf, config=self.config),
                batch_size=self.config["experiment"]["batch_size"], shuffle=True, num_workers=self.config["experiment"]["num_workers"], pin_memory=False, drop_last=True)

        self.valid_loader = DataLoader(
            AST_AudioDataset(self.data_val, audio_conf=self.val_audio_conf, config=self.config),
            batch_size=self.config["experiment"]["batch_size"] * 2, shuffle=False, num_workers=self.config["experiment"]["num_workers"], pin_memory=False)

        print('Now train with {:s} with {:d} training samples, evaluate with {:d} samples'.format(self.dataset, len(self.train_loader.dataset), len(self.valid_loader.dataset)))


    def set_model(self):
        ''' Setup e_crnn model and optimizer '''
        # Model
        self.model = ASTModel(label_dim=self.n_class,
                            fshape=self.fshape,
                            tshape=self.tshape,
                            fstride=self.fstride,
                            tstride=self.tstride,
                            input_fdim=self.num_mel_bins,
                            input_tdim=self.target_length,
                            model_size=self.model_size,
                            pretrain_stage=False,
                            load_pretrained_mdl_path=self.config["path"]["pretrained_mdl_path"],
                            hidden_layer_dim=self.hidden_layer_dim).to(self.device)

        # fix layers except last layer (mlp_head)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.mlp_head.parameters():
            param.requires_grad = True
        
        self.verbose("number of trainable parameters: {}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.reg_loss_func = nn.MSELoss() # regression loss
        self.rank_loss_func = nn.BCELoss() # ranking loss
        

        # Optimizer
        self.optimizer = Optimizer(self.model.parameters(), **self.config['hparas'])
        self.verbose(self.optimizer.create_msg())

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt(cont=True)

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
            
            self.model.load_state_dict(ckpt['model'])

            # Load task-dependent items
            if self.mode == 'train':
                if cont:
                    self.step = ckpt['global_step']
                    self.epoch = ckpt['global_epoch']
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

            if self.use_ranking_loss:
                raise Exception ("Not implemented yet")

                for i, data in enumerate(tqdm(self.train_loader)):
                    # Pre-step : update tf_rate/lr_rate and do zero_grad
                    # tf_rate = self.optimizer.pre_step(self.step)
                    self.optimizer.opt.zero_grad()
                    total_loss = 0

                    # Fetch data
                    img_1, img_2, lab_scores_1, lab_scores_2 = self.fetch_data(data)
                    self.timer.cnt('rd')
                    lab_scores = torch.cat((lab_scores_1, lab_scores_2))


                    # Forward model
                    pred_scores_1 = self.model(img_1)
                    pred_scores_2 = self.model(img_2)
                    pred_scores = torch.cat((pred_scores_1, pred_scores_2))

                    reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    # reg_loss = torch.sqrt(reg_loss)
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
                for i, data in enumerate(tqdm(self.train_loader)):
                    # Pre-step : update tf_rate/lr_rate and do zero_grad
                    # tf_rate = self.optimizer.pre_step(self.step)
                    self.optimizer.opt.zero_grad()
                    total_loss = 0

                    fbanks, lab_scores = self.fetch_data(data)
                    self.timer.cnt('rd')

                    # Forward model
                    pred_scores = self.model(fbanks, task="ft_avgtok")
                    reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    # reg_loss = torch.sqrt(reg_loss)
                    train_reg_prediction.append(pred_scores)
                    train_reg_loss.append(reg_loss.cpu().detach().numpy())
                    
                    total_loss = reg_loss
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

        self.log.close()

    def validate(self):
        # Eval mode
        self.model.eval()
        valid_reg_loss, valid_rank_loss, valid_total_loss = [], [], [] # record the loss of every batch
        valid_reg_prediction, valid_rank_prediction = [], []

        for i, data in enumerate(self.valid_loader):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.valid_loader)))
            if self.use_ranking_loss:
                raise Exception ("Not implemented yet")
                # Fetch data
                img_1, img_2, lab_scores_1, lab_scores_2 = self.fetch_data(data)
                lab_scores = torch.cat((lab_scores_1, lab_scores_2))

                # Forward model
                with torch.no_grad():
                    pred_scores_1 = self.model(img_1)
                    pred_scores_2 = self.model(img_2)
                    pred_scores = torch.cat((pred_scores_1, pred_scores_2))

                    valid_reg_prediction.append(pred_scores)
                    reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    # reg_loss = torch.sqrt(reg_loss)
                    valid_reg_loss.append(reg_loss.cpu().detach().numpy())

                    valid_rank_prediction.append(pred_scores_1 > pred_scores_2)
                    pred_binary_rank = nn.Sigmoid()(pred_scores_1 - pred_scores_2)
                    lab_binary_rank = (pred_scores_1>pred_scores_2).float().to(self.device)
                    rank_loss = self.rank_loss_func(pred_binary_rank, lab_binary_rank)
                    valid_rank_loss.append(rank_loss.cpu().detach().numpy())

                    total_loss = reg_loss + self.ranking_weight*rank_loss
                    valid_total_loss.append(total_loss.cpu().detach().numpy())
            else:
                fbanks, lab_scores = self.fetch_data(data)
                with torch.no_grad():
                    pred_scores = self.model(fbanks, task="ft_avgtok")
                    reg_loss = self.reg_loss_func(pred_scores, torch.unsqueeze(lab_scores, 1))
                    # reg_loss = torch.sqrt(reg_loss)
                    valid_reg_prediction.append(pred_scores)
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
            self.write_log('MSE_loss', {'total_loss/valid': epoch_valid_total_loss})


        # Ckpt if performance improves
        
        if epoch_valid_total_loss < self.best_valid_loss:
            self.best_valid_loss = epoch_valid_total_loss
            self.save_checkpoint('{}_best.pth'.format(
                self.paras.model), 'total_loss', epoch_valid_total_loss)

        # Resume training
        self.model.train()
        return epoch_valid_total_loss
