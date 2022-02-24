import os
import yaml
import math
import json
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.solver import BaseSolver
from src.optim import Optimizer
from models.pase_model import wf_builder
from models.memorability_model import LSTM
from models.probing_model import Probing_Model
from src.dataset import MemoWavDataset, ReconstructionDataset
from src.util import human_format, get_grad_norm
from torch.utils.data import DataLoader

CKPT_STEP = 10000
SAMPLING_RATE = 16000

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.log_freq = self.config["experiment"]['log_freq']
        self.best_valid_loss = float('inf')
        self.memo_model = self.config["experiment"]["memo_model"]
        self.CUDA = True if self.paras.gpu else False
        assert self.memo_model == "pase_lstm", "only pase_lstm is supported for now"
        self.save_hidden_states = self.config["experiment"]["store_hidden_states"]

    def fetch_data(self, data):
        ''' Move data to device '''

        hidden_states, mels = data
        hidden_states, mels = hidden_states.to(self.device).float(), mels.to(self.device).float()

        return hidden_states, mels

    def store_hidden_states(self):
        self.verbose("Storing hidden states")
        
        with open("config/{}.yaml".format(self.memo_model), 'r') as f:
            self.memo_config = yaml.load(f, Loader=yaml.FullLoader)
        options = self.memo_config["model"]
        inp_dim = options["input_dim"]

        with open(self.memo_config["path"]["fe_cfg"], 'r') as fe_cfg_f:
            self.fe_cfg = json.load(fe_cfg_f)
        encoder = wf_builder(self.fe_cfg).to(self.device)


        if self.memo_model == "pase_lstm":    
            dataset = MemoWavDataset(self.labels_df,
                                    self.memo_config["path"]["data_root"][0],
                                    self.memo_config["path"]["data_cfg"][0], 
                                    'all',
                                    sr=SAMPLING_RATE,
                                    preload_wav=self.memo_config["dataset"]["preload_wav"],
                                    same_sr=True)
        
            downstream_model = LSTM(options=options, inp_dim=inp_dim).to(self.device)

            model_ckpt = torch.load(self.config["path"]["model_ckpt_file"], map_location=self.device)
        else:
            raise Exception ("Only pase_lstm is supported for now")

        encoder.load_state_dict(model_ckpt["encoder"])
        downstream_model.load_state_dict(model_ckpt["model"])

        self.verbose("dataset length: {}".format(len(dataset)))

        # loop over all tracks
        for track_idx in tqdm(range(len(dataset))):
            track = self.labels_df.track.values[track_idx]
            # loop over all augmented versions of the track
            for fname in dataset.filename_options[track_idx]:
                
                augmented_type = fname.split('/')[-2]
                hidden_states_subdir = os.path.join(self.config["path"]["hidden_states_dir"], track, augmented_type)
                os.makedirs(hidden_states_subdir, exist_ok=True)

                wav = dataset.retrieve_cache(fname, dataset.wav_cache)
                # need to convert to tensor manually without using the dataloader
                wav = torch.tensor(wav).unsqueeze(0)
                wav = wav.view(wav.size(0), 1, -1).to(self.device)
                features = encoder(wav, self.device)
                pred_score, hidden_states = downstream_model(features)
                
                for layer_index, layer_input in enumerate(hidden_states):
                    layer_input = layer_input.detach().cpu().numpy()
                    np.save(os.path.join(hidden_states_subdir, "{}.npy".format(layer_index)), layer_input)


        self.verbose("hidden states stored at {}".format(self.config["path"]["hidden_states_dir"]))

    def load_data(self):
        ''' Load data for training/validation '''
        
        self.labels_df = pd.read_csv(self.config["path"]["label_file"])
        if self.save_hidden_states:
            self.store_hidden_states()


        # indexing except testing indices
        fold_size = int(len(self.labels_df) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        for_test = self.labels_df.index.isin(testing_range)
        self.labels_df = self.labels_df[~for_test]
        self.labels_df = self.labels_df.sample(frac=1, random_state=self.paras.seed).reset_index(drop=True)
        self.valid_labels_df = self.labels_df[:fold_size].reset_index(drop=True)
        self.train_labels_df = self.labels_df[fold_size:].reset_index(drop=True)

        self.train_set = ReconstructionDataset(self.train_labels_df,
                                                self.config,
                                                "train")
        self.valid_set = ReconstructionDataset(self.valid_labels_df,
                                                self.config,
                                                "valid")
        self.verbose("train dataset length: {}".format(len(self.train_set)))
        self.verbose("valid dataset length: {}".format(len(self.valid_set)))

        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.config["experiment"]["batch_size"],
                            num_workers=self.config["experiment"]["num_workers"], shuffle=True)
        self.valid_loader = DataLoader(dataset=self.valid_set, batch_size=self.config["experiment"]["batch_size"],
                            num_workers=self.config["experiment"]["num_workers"], shuffle=False)
        

    def set_model(self):
        ''' Setup h_mlp model and optimizer '''
        # Model
        self.verbose("Setting up probbing model")
        self.model = Probing_Model(model_config=self.config["model"]).to(self.device)

        # Losses
        self.verbose("Using {} as loss function".format(self.model.Loss_Function))
        

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
        ''' Training probing '''
        self.verbose('Total training steps {}.'.format(
            human_format(self.max_step)))
        self.timer.set()

        for self.epoch in range(self.max_epoch):
            print("\nepoch: {}/{}".format(self.epoch+1, self.max_epoch))
            
            self.model.train()
            train_total_loss = [] # record the loss of every batch
                        
            for i, data in enumerate(tqdm(self.train_loader)):
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                # tf_rate = self.optimizer.pre_step(self.step)
                self.optimizer.opt.zero_grad()

                # Fetch data
                hidden_states, mels = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward model
                predictions, loss = self.model(hidden_states, mels)

                train_total_loss.append(loss.cpu().detach().numpy())
                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(loss)
                self.step += 1

                if i % self.log_freq == 0:
                    self.log.add_scalars('train_loss', {'total_loss/train': np.mean(train_total_loss)}, self.step)


            # Logger
            self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
                            .format(loss.cpu().item(), grad_norm, self.timer.show()))

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
        valid_total_loss = [] # record the loss of every batch

        for idx, data in enumerate(self.valid_loader):
            self.progress('Valid step - {}/{}'.format(idx+1, len(self.valid_loader)))
            # Fetch data
            hidden_states, mels = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                predicted_mels, loss = self.model(hidden_states, mels)
                valid_total_loss.append(loss.cpu().detach().numpy())

                if idx%40 == 0:
                    # (N, n_mels, downsmapling_factor) 
                    mels = mels.permute(1,0,2)
                    # (n_mels, N, downsmapling_factor) 
                    mels = mels.reshape(-1, mels.size(1)*mels.size(2))
                    # (n_mels, N*downsmapling_factor) 
                    mels = mels.cpu().detach().numpy()
                    fig = self.convert_mels_to_fig(mels)
                    self.log.add_figure("{}/valid_label".format(idx), fig, self.epoch)

                    # (N, n_mels, downsmapling_factor) 
                    predicted_mels = predicted_mels.permute(1,0,2)
                    # (n_mels, N, downsmapling_factor) 
                    predicted_mels = predicted_mels.reshape(-1, predicted_mels.size(1)*predicted_mels.size(2))
                    # (n_mels, N*downsmapling_factor) 
                    predicted_mels = predicted_mels.cpu().detach().numpy()
                    fig = self.convert_mels_to_fig(predicted_mels)
                    self.log.add_figure("{}/valid_pred".format(idx), fig, self.epoch)


        epoch_valid_total_loss = np.mean(valid_total_loss)
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
        self.model.train()
        return epoch_valid_total_loss

    def convert_mels_to_fig(self, predicted_mels):
        ''' return figure (from matplotlib) of melspectrogram '''
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set(title="Log-Power spectrogram")
        S_dB = librosa.power_to_db(predicted_mels, ref=np.max)
        p = librosa.display.specshow(S_dB, ax=ax, y_axis='log', x_axis='time')
        fig.colorbar(p, ax=ax, format="%+2.0f dB")

        return fig

