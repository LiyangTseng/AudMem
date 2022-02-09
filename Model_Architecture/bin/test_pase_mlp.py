import os
import json
import csv
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from scipy import stats
from src.solver import BaseSolver
from models.memorability_model import MLP
from models.pase_model import wf_builder
from src.dataset import MemoWavDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

SAMPLING_RATE = 16000

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        output_dir = os.path.join(self.outdir, paras.model)
        os.makedirs(output_dir, exist_ok=True)
        self.memo_output_path = os.path.join(output_dir, "predicted_memorability_scores.csv")
        self.corr_output_path = os.path.join(output_dir, "details.txt")
        self.interp_dir = os.path.join(self.outdir, paras.model, "interpretability")
        
        with open(self.config["path"]["fe_cfg"], 'r') as fe_cfg_f:
            self.fe_cfg = json.load(fe_cfg_f)

        self.CUDA = True if self.paras.gpu else False
        self.model_ckpt = torch.load(self.config["path"]["finetuned_model_ckpt"], map_location=self.device)
        self.verbose("Loaded model checkpoint @ {}".format(self.config["path"]["finetuned_model_ckpt"]))

    def load_data(self):
        ''' Load data for testing '''

        # get labels from csv file
        self.labels_df = pd.read_csv(self.config["path"]["label_file"])
        # construct filename: score, ref: https://stackoverflow.com/questions/18012505/python-pandas-dataframe-columns-convert-to-dict-key-and-value
        self.filename_to_score = dict(zip(self.labels_df.track, self.labels_df.score))

        self.test_set = MemoWavDataset(self.filename_to_score,
                                    self.config["path"]["data_root"][0],
                                    self.config["path"]["data_cfg"][0], 
                                    'test',
                                    sr=SAMPLING_RATE,
                                    preload_wav=self.config["dataset"]["preload_wav"],
                                    same_sr=True)

        self.bpe = (self.test_set.total_wav_dur // self.config["dataset"]["chunk_size"]) // self.config["experiment"]["batch_size"]
        self.verbose("Test data length: {}".format(self.test_set.total_wav_dur/16000/3600.0))

        
        self.test_loader = DataLoader(self.test_set, batch_size=1,
                                        shuffle=False,
                                        num_workers=self.config["experiment"]["num_workers"],drop_last=True,
                                        pin_memory=self.CUDA)
        
    def fetch_data(self, data):
        ''' Move data to device '''
        wavs, scores = data
        wavs, scores = wavs.view(wavs.size(0), 1, -1).to(self.device), scores.to(self.device).float()

        return wavs, scores

    def set_model(self):
        ''' Setup downstream memorability model '''

        self.encoder = wf_builder(self.fe_cfg).to(self.device)
        self.encoder.load_state_dict(self.model_ckpt["encoder"])

        # Downstream Model
        options = self.config["model"]
        print(options)        
        inp_dim = options["input_dim"]

        self.downstream_model = MLP(options,inp_dim).to(self.device)
        self.downstream_model.load_state_dict(self.model_ckpt["model"])

        self.reg_loss_func = nn.MSELoss()
        

    def exec(self):
        ''' Testing Memorabiliy Regression/Ranking System '''

        self.pred_scores = []

        with open(self.memo_output_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["track", "pred_score", "lab_score"])
            for idx, data in enumerate(tqdm(self.test_loader)):
                wavs, lab_scores = self.fetch_data(data)
                features = self.encoder(wavs, self.device)
                features = torch.mean(features, dim=1) # temporal pooling (B, T, D) -> (B, D)
                pred_scores = self.downstream_model(features).cpu().detach().item()

                self.pred_scores.append(pred_scores)
                writer.writerow([self.labels_df.track.values[220+idx], pred_scores, self.labels_df.score.values[220+idx]])
        
            self.verbose("predicted memorability score saved at {}".format(self.memo_output_path))
        
        prediction_df = pd.read_csv(self.memo_output_path)
        correlation = stats.spearmanr(prediction_df.pred_score.values, self.labels_df.score.values[220:])
        reg_loss = torch.nn.MSELoss()(torch.tensor(prediction_df.pred_score.values).unsqueeze(0), torch.tensor(self.labels_df.score.values[220:]).unsqueeze(0))

        with open(self.corr_output_path, 'w') as f:
            f.write(str(correlation))
            f.write("regression loss: {}".format(str(correlation)))

        self.verbose("correlation result: {}, MSE loss: {}, saved at {}".format(correlation, reg_loss, self.corr_output_path))
        
        # TODO:
        # self.interpret_model()

    def interpret_model(self, N=5):
        ''' Use Captum to interprete feature importance on top N memorability score '''
        
        # ref: https://github.com/pytorch/captum/issues/564
        torch.backends.cudnn.enabled=False
        ig = IntegratedGradients(self.model)

        # ref: https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        sorted_score_idx = [idx for score, idx in sorted(zip(self.pred_scores, [i for i in range(len(self.test_set))]), reverse=True)]
        
        for idx in sorted_score_idx[:N]:
            data = (feat.unsqueeze(0) for feat in self.test_set[idx])
            feat = self.fetch_data(data)
            attributes = ig.attribute(feat, n_steps=2000)
            sns.heatmap(attributes.squeeze(0).cpu().detach().numpy())
            interp_path = os.path.join(self.interp_dir, "heatmap_"+self.test_set.idx_to_filename[220+idx].replace(".wav", ".png"))
            plt.savefig(interp_path)



