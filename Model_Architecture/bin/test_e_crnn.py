import os
import csv
from captum.attr._utils import attribution
import torch
import pandas as pd
from tqdm import tqdm
from scipy import stats
from src.solver import BaseSolver
from models.memorability_model import E_CRNN, CRNN
from src.dataset import EndToEndImgDataset, AudioDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, visualization

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        
        self.memo_output_path = os.path.join(self.outdir, "predicted_memorability_scores.csv")
        self.corr_output_path = os.path.join(self.outdir, "details.txt")
        
        self.interp_dir = os.path.join(self.outdir, paras.model, "interpretability")        
        os.makedirs(self.interp_dir, exist_ok=True)

    def fetch_data(self, data):
        ''' Move data to device '''
        mels_img, labeled_scores = data
        mels_img, labeled_scores = mels_img.to(self.device), labeled_scores.to(self.device)

        return mels_img, labeled_scores


    def load_data(self):
        ''' Load data for testing '''
        # self.test_set = EndToEndImgDataset(config=self.config, mode="test")
        self.labels_df = pd.read_csv(self.config["path"]["label_file"])
        
        fold_size = int(len(self.labels_df) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        for_test = self.labels_df.index.isin(testing_range)
        self.test_labels_df = self.labels_df[for_test].reset_index(drop=True)
        self.test_set = AudioDataset(labels_df=self.test_labels_df, config=self.config, split="test")
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=1,
                            num_workers=self.config["experiment"]["num_workers"], shuffle=False)
        
        data_msg = ('I/O spec.  | visual feature = {}\t| image shape = ({},{})\t'
                .format("melspectrogram", self.config["model"]["image_size"][0], self.config["model"]["image_size"][1]))

        self.verbose(data_msg)

    def set_model(self):
        ''' Setup e_crnn model and optimizer '''
        # Model
        # self.model = E_CRNN(model_config=self.config["model"]).to(self.device)
        self.model = CRNN(imgH=self.config["model"]["image_size"][0], \
                        nc=self.config["model"]["nc"], \
                        nclass=self.config["model"]["nclass"], \
                        nh=self.config["model"]["nh"], \
                        n_rnn=self.config["model"]["n_rnn"], \
                        leakyRelu=self.config["model"]["leakyRelu"]).to(self.device)
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
                mels_img, lab_scores = self.fetch_data(data)
                pred_score = self.model(mels_img).cpu().detach().item()
                self.pred_scores.append(pred_score)
                writer.writerow([self.test_labels_df.track.values[idx], pred_score, self.test_labels_df.score.values[idx]])

        
            self.verbose("predicted memorability score saved at {}".format(self.memo_output_path))
        
        prediction_df = pd.read_csv(self.memo_output_path)
        correlation = stats.spearmanr(prediction_df.pred_score.values, self.test_labels_df.score.values)
        reg_loss = torch.nn.MSELoss()(torch.tensor(prediction_df.pred_score.values).unsqueeze(0), torch.tensor(self.test_labels_df.score.values).unsqueeze(0))

        with open(self.corr_output_path, 'w') as f:
            f.write("using weight: {}\n".format(self.paras.load))
            f.write(str(correlation)+"\n")
            f.write("regression loss: {}\n".format(str(reg_loss)))

        self.verbose("correlation result: {}, regression loss: {}, saved at {}".format(correlation, reg_loss, self.corr_output_path))
        # self.interpret_model()

    def interpret_model(self, N=5):
        ''' Use Captum to interprete feature importance on top N memorability score '''
        
        # ref: https://github.com/pytorch/captum/issues/564
        torch.backends.cudnn.enabled=False
        ig = IntegratedGradients(self.model)

        # ref: https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        sorted_score_idx = [idx for score, idx in sorted(zip(self.pred_scores, [i for i in range(len(self.test_set))]), reverse=True)]
        
        for idx in tqdm(sorted_score_idx[:N]):
            mels_img, _ = self.fetch_data(self.test_set[idx])
            attributes = ig.attribute(mels_img.unsqueeze(0))
            # move channel to last dimesion to fit captum format
            attributes = attributes.squeeze(0).permute(1,2,0).cpu().detach().numpy()
            origin_image = mels_img.permute(1,2,0).cpu().detach().numpy()
            fig, ax = visualization.visualize_image_attr(attributes, original_image=origin_image, method="blended_heat_map", show_colorbar=True)
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            ax.axis('tight')
            ax.axis('off')


            # sns.heatmap(attributes[0].squeeze(0).cpu().detach().numpy().T)
            interp_path = os.path.join(self.interp_dir, "heatmap_"+self.test_set.idx_to_filename[220+idx].replace(".wav", ".png"))
            plt.savefig(interp_path)
            plt.close()
        
        self.verbose("interpretable feature heat map saved at {}".format(self.interp_dir))






