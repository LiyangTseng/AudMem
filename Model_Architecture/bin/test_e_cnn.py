import os
import csv
from captum.attr._utils import attribution
import torch
import pandas as pd
from tqdm import tqdm
from scipy import stats
from src.solver import BaseSolver
from models.memorability_model import CNN
from src.dataset import SoundDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, visualization

from lime import lime_image
import numpy as np
from skimage.segmentation import mark_boundaries
import librosa
from moviepy.editor import *
from moviepy.video.io.bindings import mplfig_to_npimage


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
        mels_img, labeled_scores, _ = data
        mels_img, labeled_scores = mels_img.to(self.device), labeled_scores.to(self.device)
            
        return mels_img, labeled_scores


    def load_data(self):
        ''' Load data for testing '''
        self.labels_df = pd.read_csv(self.config["path"]["label_file"])
        
        fold_size = int(len(self.labels_df) / self.paras.kfold_splits)
        testing_range = [ i for i in range(self.paras.fold_index*fold_size, (self.paras.fold_index+1)*fold_size)]
        for_test = self.labels_df.index.isin(testing_range)
        self.test_labels_df = self.labels_df[for_test].reset_index(drop=True)
        self.test_set = SoundDataset(labels_df=self.test_labels_df, config=self.config, split="test")
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=1,
                            num_workers=self.config["experiment"]["num_workers"], shuffle=False)
        
    def set_model(self):
        ''' Setup e_crnn model and optimizer '''
        # Model
        self.model = CNN().to(self.device)

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
                # mels_img shape: (batch_size=1, channel=1, n_mels=64, time_steps=431)
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
        self.interpret_model()

    def interpret_model(self, N=5):
        ''' Use Captum to interprete feature importance on top N memorability score '''
        
        # # ref: https://github.com/pytorch/captum/issues/564
        # torch.backends.cudnn.enabled=False
        # ig = IntegratedGradients(self.model)

        self.explainer = lime_image.LimeImageExplainer()
        
        # # ref: https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        sorted_score_idx = [idx for score, idx in sorted(zip(self.pred_scores, [i for i in range(len(self.test_set))]), reverse=True)]
        cnt = 0
        for idx in sorted_score_idx[:N]:
            self.verbose("generating lime explanation {}/{}".format(cnt+1, N))
            mels_img, _ = self.fetch_data(self.test_set[idx])
            # mels_img shape: (channel=1, n_mels=64, time_steps=431)
            self.save_explanatory_video(mels_img, idx)
            cnt += 1

        self.verbose("interpretable feature heat map saved at {}".format(self.interp_dir))


    def save_explanatory_video(self, spec_input, idx):
        
        n_fft = 1024
        hop_length = 512
        sr = 44100
        mels_ticks = [0, 20, 40, 60]
        seconds = 5
        # frame_ticks = [0, 50, 100, 150, 200, 250, 300, 350, 400]
        time_steps = int(seconds * sr * (n_fft/hop_length) / n_fft) + 1
        frame_ticks = [i*time_steps/5 for i in range(5)]
        def wrapped_net(x):
            if x.shape[-1] == 3:
                x = x[:, :, :, 0] # make it one-channel
            x = torch.tensor(x).unsqueeze(0).to(self.device).float()
            return self.model(x).cpu().detach().numpy()

        # tmp = np.array(spec_input.squeeze()[0].double().cpu())
        # original_spec = np.stack((tmp,tmp,tmp), axis=-1)
        original_spec = spec_input.squeeze().double().cpu()
        # normalize to 0-1
        original_spec = (original_spec - original_spec.min()) / (original_spec.max() - original_spec.min())
        exp = self.explainer.explain_instance(image=original_spec, classifier_fn=wrapped_net, batch_size=1)
    
        temp, mask = exp.get_image_and_mask(0, positive_only=True, 
                                num_features=5, hide_rest=True)
        img_boundry1 = mark_boundaries(temp, mask)    
        img_boundry1 = (img_boundry1 - img_boundry1.min()) / (img_boundry1.max() - img_boundry1.min())
        
        # plot spectrogram input, explanation, and audio waveform
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 8))
        plt.setp((ax1, ax2), xticks=frame_ticks, xticklabels=["{:.1f}".format((frame)/(n_fft/hop_length)*n_fft/sr)  for frame in frame_ticks],
        yticks=mels_ticks, yticklabels=[int(librosa.mel_frequencies(n_mels=64)[mel])  for mel in mels_ticks])
        fig.text(0.5, 0.04, 'Time [s]', ha='center', fontsize=11)
        
        # plot spectrogram
        ax1.imshow(original_spec)
        ax1.set_title('Original')
        ax1.invert_yaxis()
        ax1.set_ylabel('Frequency [Hz]')
        
        # plot explanation
        ax2.imshow(img_boundry1)
        ax2.set_title('Lime interpretability')
        ax2.invert_yaxis()
        ax2.set_ylabel('Frequency [Hz]')
        
        # plot audio waveform
        audio_path = os.path.join(self.test_set.audio_root, self.test_set.audio_dirs[0], self.test_set.track_names[idx])
        audio, sr = librosa.load(audio_path, sr=sr)
        line,v = ax3.plot(0, audio.min(), 5*sr, audio.max(), linewidth=2, color='black')
        ax3.plot(audio, linewidth=2)

        sampling_ticks = [i*audio.shape[0]/5 for i in range(5)]
        plt.setp((ax3), xticks=sampling_ticks, xticklabels=["{:.1f}".format(frame/sr)  for frame in sampling_ticks])
        ax3.margins(x=0)
        ax3.set_title('Audio')
        ax3.set_ylabel('Amplitude')


        # save to video
        self.verbose("saving explanations to video")
        audioclip = AudioFileClip(audio_path)
        new_audioclip = CompositeAudioClip([audioclip])

        frames = 50
        X_VALS = np.linspace(0, 5, frames)

        def make_frame(t):
            x = X_VALS[int(t*frames/5)]
            # ax3.plot([x*sr, x*sr], [audio.min(), audio.max()], color='red', linewidth=2)
            line.set_data( [x*sr, x*sr], [audio.min(), audio.max()])

            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration=audioclip.duration)

        animation.audio = new_audioclip

        interp_path = os.path.join(self.interp_dir, "lime_"+self.test_set.track_names[idx].replace(".wav", ".mp4"))
        animation.write_videofile(interp_path, fps=frames/5, audio_codec='aac')
        self.verbose("lime video saved at {}".format(interp_path))







