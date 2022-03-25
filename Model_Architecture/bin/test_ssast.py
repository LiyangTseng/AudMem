import os
import csv
from captum.attr._utils import attribution
import torch
import pandas as pd
from tqdm import tqdm
from scipy import stats
from src.solver import BaseSolver
from models.ast_models import ASTModel
from src.dataset import AST_AudioDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, visualization

class Solver(BaseSolver):
    ''' Solver for training'''

    def parse_yaml(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                self.parse_yaml(value)
            else:
                setattr(self, key, value)


    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        output_dir = os.path.join(self.outdir, paras.model)
        os.makedirs(output_dir, exist_ok=True)
        self.memo_output_path = os.path.join(output_dir, "predicted_memorability_scores.csv")
        self.corr_output_path = os.path.join(output_dir, "details.txt")
        
        self.parse_yaml(self.config)

        self.test_audio_conf = {'num_mel_bins': self.num_mel_bins, 'target_length': self.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': self.dataset,
                        'mode': 'evaluation', 'mean': self.dataset_mean, 'std': self.dataset_std, 'noise': False}


        self.interp_dir = os.path.join(self.outdir, paras.model, "interpretability")        
        os.makedirs(self.interp_dir, exist_ok=True)

    def fetch_data(self, data):
        ''' Move data to device '''
        fbank, label = data
        fbank = fbank.to(self.device)
        label = label.to(self.device).float()
        return fbank, label


    def load_data(self):
        ''' Load data for testing '''
        # self.test_set = EndToEndImgDataset(config=self.config, mode="test")
        self.labels_df = pd.read_csv(self.config["path"]["label_file"])
        
        fold_size = int(len(self.labels_df) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        for_test = self.labels_df.index.isin(testing_range)
        self.test_labels_df = self.labels_df[for_test].reset_index(drop=True)
        
        self.test_loader = DataLoader(
            AST_AudioDataset(self.data_eval, audio_conf=self.test_audio_conf, config=self.config),
            batch_size=1, shuffle=False, num_workers=self.config["experiment"]["num_workers"], pin_memory=False)

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
                            load_pretrained_mdl_path=self.paras.load,
                            hidden_layer_dim=self.hidden_layer_dim).to(self.device)

        # Load target model in eval mode
        self.load_ckpt()


    def exec(self):
        ''' Testing Memorabiliy Regression/Ranking System '''

        self.pred_scores = []
        
        with open(self.memo_output_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["track", "pred_score", "lab_score"])
            for idx, data in enumerate(tqdm(self.test_loader)):
                fbanks, lab_scores = self.fetch_data(data)
                pred_score = self.model(fbanks, task="ft_avgtok").cpu().detach().item()
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






