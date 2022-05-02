import os
import csv
import torch
import pandas as pd
from tqdm import tqdm
from scipy import stats
from src.solver import BaseSolver
from models.memorability_model import H_LSTM
from src.dataset import HandCraftedDataset
from utils.calculate_handcrafted_features_stats import get_features_stats
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        
        self.memo_output_path = os.path.join(self.outdir, "predicted_memorability_scores.csv")
        self.corr_output_path = os.path.join(self.outdir, "details.txt")
        
        self.interp_dir = os.path.join(self.outdir, "interpretability")        
        os.makedirs(self.interp_dir, exist_ok=True)

    def fetch_data(self, data):
        ''' Move data to device '''
        (seq_feats, non_seq_feats), lab_scores, weights = data
        seq_feats, non_seq_feats = seq_feats.to(self.device), non_seq_feats.to(self.device)

        return seq_feats, non_seq_feats

    def load_data(self):
        ''' Load data for testing '''
        # get labels from csv file
        self.labels_df = pd.read_csv(self.config["path"]["label_file"])
        
        fold_size = int(len(self.labels_df) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        for_test = self.labels_df.index.isin(testing_range)
        self.test_labels_df = self.labels_df[for_test].reset_index(drop=True)
        stats_dict = get_features_stats(label_df=self.test_labels_df, 
                                        features_dir=self.config["path"]["features_dir"], 
                                        for_test=True,
                                        features_dict = self.config["features"])

        self.test_set = HandCraftedDataset(labels_df=self.test_labels_df, config=self.config, stats_dict=stats_dict, pooling=False, split="test")
        
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=1,
                            num_workers=self.config["experiment"]["num_workers"], shuffle=False)
        
        data_msg = ('I/O spec.  | audio feature = {}\t| sequential feature dim = {}\t| nonsequential feature dim = {}\t'
                .format(self.test_set.features_dict, self.test_set[0][0][0].shape, self.test_set[0][0][1].shape))

        self.verbose(data_msg)

    def set_model(self):
        ''' Setup ASR model '''
        # Model
        self.model = H_LSTM(model_config=self.config["model"]).to(self.device)
        self.verbose(self.model.create_msg())

        # Load target model in eval mode
        self.load_ckpt()


    def exec(self):
        ''' Testing Memorabiliy Regression/Ranking System '''

        self.pred_scores = []

        with open(self.memo_output_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["track", "pred_score", "lab_score"])
            for idx, data in enumerate(tqdm(self.test_loader)):
                seq_feat, non_seq_feat = self.fetch_data(data)
                pred_scores = self.model(seq_feat, non_seq_feat).cpu().detach().item()
                self.pred_scores.append(pred_scores)
                writer.writerow([self.test_labels_df.track.values[idx], pred_scores, self.test_labels_df.score.values[idx]])
        
            self.verbose("predicted memorability score saved at {}".format(self.memo_output_path))
        
        prediction_df = pd.read_csv(self.memo_output_path)
        correlation = stats.spearmanr(prediction_df.pred_score.values, self.test_labels_df.score.values)
        reg_loss = torch.nn.MSELoss()(torch.tensor(prediction_df.pred_score.values).unsqueeze(0), torch.tensor(self.test_labels_df.score.values).unsqueeze(0))
        with open(self.corr_output_path, 'w') as f:
            f.write(str(correlation) + "\n")
            f.write("MSE loss: {}\n".format(str(reg_loss)))

        self.verbose("correlation result: {}, regression loss: {}, saved at {}".format(correlation, reg_loss, self.corr_output_path))

        # TODO:
        # self.interpret_model()

    def interpret_model(self, N=5):
        ''' Use Captum to interprete feature importance on top N memorability score '''
        
        # ref: https://github.com/pytorch/captum/issues/564
        torch.backends.cudnn.enabled=False
        ig = IntegratedGradients(self.model)

        # ref: https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        sorted_score_idx = [idx for score, idx in sorted(zip(self.pred_scores, [i for i in range(len(self.test_set))]), reverse=True)]
        
        for idx in tqdm(sorted_score_idx[:N]):
            data = (feat.unsqueeze(0) for feat in self.test_set[idx])
            feat = self.fetch_data(data)
            attributes = ig.attribute(feat)
            sns.heatmap(attributes[0].squeeze(0).cpu().detach().numpy().T)
            interp_path = os.path.join(self.interp_dir, "heatmap_"+self.test_set.idx_to_filename[idx].replace(".wav", ".png"))
            plt.savefig(interp_path)
            plt.close()
        
        self.verbose("interpretable feature heat map saved at {}".format(self.interp_dir))



