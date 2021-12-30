import os
import csv
import pandas as pd
from tqdm import tqdm
from scipy import stats
from src.solver import BaseSolver
from model.memorability_model import H_LSTM
from dataset import HandCraftedDataset
from torch.utils.data import DataLoader

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        self.memo_output_path = os.path.join(self.paras.outdir, "score", "predicted_memorability_scores.csv")
        self.corr_output_path = os.path.join(self.paras.outdir, "score", "correlations.txt")
        # self.ranking_weight = config["model"]["ranking_weight"]
        
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
        self.model = H_LSTM(model_config=self.config["model"], device=self.device).to(self.device)
        self.verbose(self.model.create_msg())

        # Load target model in eval mode
        self.load_ckpt()


    def exec(self):
        ''' Testing Memorabiliy Regression/Ranking System '''

        with open(self.memo_output_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["track", "score"])
            for idx, data in enumerate(tqdm(self.test_loader)):
                seq_feat, non_seq_feat = self.fetch_data(data)
                pred_scores = self.model(seq_feat, non_seq_feat)
                # pred_scores = self.model(data)
                writer.writerow([self.test_set.idx_to_filename[idx], pred_scores.cpu().detach().item()])
        
            self.verbose("predicted memorability score saved at {}".format(self.memo_output_path))
        
        prediction_df = pd.read_csv(self.memo_output_path)
        correlation = stats.spearmanr(prediction_df["score"].values, self.test_set.filename_memorability_df["score"].values)
        
        with open(self.corr_output_path, 'w') as f:
            f.write(str(correlation))
        self.verbose("correlation result: {}, saved at {}".format(correlation, self.corr_output_path))



