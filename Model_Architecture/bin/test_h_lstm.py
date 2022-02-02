import os
import csv
import torch
import pandas as pd
from tqdm import tqdm
from scipy import stats
from src.solver import BaseSolver
from models.memorability_model import H_LSTM
from src.dataset import HandCraftedDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        output_dir = os.path.join(self.outdir, paras.model)
        os.makedirs(output_dir, exist_ok=True)
        self.memo_output_path = os.path.join(output_dir, "predicted_memorability_scores.csv")
        self.corr_output_path = os.path.join(output_dir, "details.txt")

        self.interp_dir = os.path.join(self.outdir, paras.model, "interpretability")        
        os.makedirs(self.interp_dir, exist_ok=True)

    def fetch_data(self, data):
        ''' Move data to device '''
        seq_feat, non_seq_feat = data
        seq_feat, non_seq_feat = seq_feat.to(self.device), non_seq_feat.to(self.device)

        return seq_feat, non_seq_feat


    def load_data(self):
        ''' Load data for testing '''
        self.test_set = HandCraftedDataset(config=self.config, pooling=False, mode="test")
        
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=1,
                            num_workers=self.config["experiment"]["num_workers"], shuffle=False)
        
        data_msg = ('I/O spec.  | audio feature = {}\t| sequential feature dim = {}\t| nonsequential feature dim = {}\t'
                .format(self.test_set.features_dict, self.test_set.sequential_features[0].shape, self.test_set.non_sequential_features[0].shape))

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
            writer.writerow(["track", "score"])
            for idx, data in enumerate(tqdm(self.test_loader)):
                seq_feat, non_seq_feat = self.fetch_data(data)
                pred_score = self.model(seq_feat, non_seq_feat)
                self.pred_scores.append(pred_score)
                writer.writerow([self.test_set.idx_to_filename[idx], pred_score.cpu().detach().item()])
        
            self.verbose("predicted memorability score saved at {}".format(self.memo_output_path))
        
        prediction_df = pd.read_csv(self.memo_output_path)
        correlation = stats.spearmanr(prediction_df["score"].values, self.test_set.filename_memorability_df["score"].values)
        reg_loss = torch.nn.MSELoss()(torch.tensor(prediction_df["score"].values).unsqueeze(0), torch.tensor(self.test_set.filename_memorability_df["score"].values).unsqueeze(0))


        with open(self.corr_output_path, 'w') as f:
            f.write("using weight: {}\n".format(self.paras.load))
            f.write(str(correlation)+"\n")
            f.write("regression loss: {}\n".format(str(correlation)))

        self.verbose("correlation result: {}, regression loss: {}, saved at {}".format(correlation, reg_loss, self.corr_output_path))
        
        # TODO:
        self.interpret_model()

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


