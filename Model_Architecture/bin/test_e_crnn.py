import os
import csv
import pandas as pd
from tqdm import tqdm
from scipy import stats
from src.solver import BaseSolver
from model.memorability_model import E_CRNN
from dataset import EndToEndImgDataset
from torch.utils.data import DataLoader

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        output_dir = os.path.join(self.outdir, paras.model, "score")
        os.makedirs(output_dir, exist_ok=True)
        self.memo_output_path = os.path.join(output_dir, "predicted_memorability_scores.csv")
        self.corr_output_path = os.path.join(output_dir, "correlations.txt")
        
        
    def fetch_data(self, data):
        ''' Move data to device '''
        mels_img, labeled_scores = data
        mels_img, labeled_scores = mels_img.to(self.device), labeled_scores.to(self.device)

        return mels_img, labeled_scores


    def load_data(self):
        ''' Load data for testing '''
        self.test_set = EndToEndImgDataset(config=self.config, mode="test")
        
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=1,
                            num_workers=self.config["experiment"]["num_workers"], shuffle=False)
        
        data_msg = ('I/O spec.  | visual feature = {}\t| image shape = ({},{})\t'
                .format("melspectrogram", self.config["model"]["image_size"], self.config["model"]["image_size"]))

        self.verbose(data_msg)

    def set_model(self):
        ''' Setup e_crnn model and optimizer '''
        # Model
        self.model = E_CRNN(model_config=self.config["model"]).to(self.device)
        self.verbose(self.model.create_msg())

        # Load target model in eval mode
        self.load_ckpt()


    def exec(self):
        ''' Testing Memorabiliy Regression/Ranking System '''

        with open(self.memo_output_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["track", "score"])
            for idx, data in enumerate(tqdm(self.test_loader)):
                mels_img, lab_scores = self.fetch_data(data)
                pred_scores = self.model(mels_img)
                writer.writerow([self.test_set.idx_to_filename[idx], pred_scores.cpu().detach().item()])
        
            self.verbose("predicted memorability score saved at {}".format(self.memo_output_path))
        
        prediction_df = pd.read_csv(self.memo_output_path)
        correlation = stats.spearmanr(prediction_df["score"].values, self.test_set.filename_memorability_df["score"].values)
        
        with open(self.corr_output_path, 'w') as f:
            f.write(str(correlation))
        self.verbose("correlation result: {}, saved at {}".format(correlation, self.corr_output_path))



