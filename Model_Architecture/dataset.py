import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HandCraftedDataset(Dataset):
    ''' Hand crafted audio features to predict memorability (classification) '''
    def __init__(self, config, pooling=False, mode="train"):
        super().__init__()
        self.features_dir = config["path"]["features_dir"]
        self.lables_dir = config["path"]["labels_dir"]
        self.augmented_type_list = sorted(os.listdir(self.features_dir))
        self.pooling = pooling
        if mode == "train":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.lables_dir, "track_memorability_scores_beta.csv"))[:200]
        elif mode == "valid":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.lables_dir, "track_memorability_scores_beta.csv"))[200:220]
        elif mode == "test":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.lables_dir, "track_memorability_scores_beta.csv"))[220:]
        elif mode == "all":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.lables_dir, "track_memorability_scores_beta.csv"))            
        else:
            raise Exception ("Invalid dataset mode")

        self.idx_to_filename = {idx: filename for idx, filename in enumerate(self.filename_memorability_df["track"])}
        self.idx_to_mem_score = {idx: score for idx, score in enumerate(self.filename_memorability_df["score"])}

        # self.features_dict = {"chords": ["chroma"], "rhythms": ["tempogram"], "timbre": ["mfcc"], "emotions": ["static_arousal", "static_valence"]}
        self.features_dict = config["features"]

    def __len__(self):
        return len(self.idx_to_filename)*len(self.augmented_type_list)       

    def __getitem__(self, index):
        
        sequeutial_features_list = []
        non_sequential_features_list = []
        
        
        for feature_type in self.features_dict:
            for subfeatures in self.features_dict[feature_type]:
                feature_file_path = os.path.join(self.features_dir, 
                self.augmented_type_list[index//len(self.idx_to_filename)], feature_type, subfeatures,
                 "{}_{}".format(subfeatures, self.idx_to_filename[index%len(self.idx_to_filename)].replace("wav", "npy")))
                
                f = np.load(feature_file_path)
                if feature_type == "emotions":
                    # features not sequential
                    non_sequential_features_list.append(f)
                else:
                    # features are sequential 
                    if self.pooling:
                        # temporal mean pooling to compress 2d features to 1d
                        sequeutial_features_list.append(f.mean(axis=1))
                        sequential_features = np.concatenate(sequeutial_features_list, axis=0)
                    else:
                        sequeutial_features_list.append(f)
                        sequential_features = np.concatenate(sequeutial_features_list, axis=0)
                        sequential_features = np.transpose(sequential_features)
  
        non_sequential_features = np.array(non_sequential_features_list)
        labels = self.idx_to_mem_score[index%len(self.idx_to_filename)]
    
        return torch.tensor(sequential_features), torch.tensor(non_sequential_features), torch.tensor(labels, dtype=torch.double)

class EndToEndDataset(Dataset):
    ''' End-to-end audio features to predict memorability (classification) '''
    pass

class ReconstructionDataset(Dataset):
    ''' LSTM　hidden states to reconstruct mel-spectrograms ref: https://arxiv.org/pdf/1911.01102.pdf '''
    def __init__(self, config):
        super().__init__()
        self.hidden_states_dir = config["path"]["hidden_states_dir"]
        self.features_dir = config["path"]["features_dir"]
        self.hidden_layer_num = 4
        self.augmented_type_list = os.listdir(self.hidden_states_dir)
        self.sequence_length = 216

        filename_memorability_df = pd.read_csv("data/labels/track_memorability_scores_beta.csv")
        self.idx_to_filename = {idx: filename for idx, filename in enumerate(filename_memorability_df["track"])}

    def __len__(self):
        return len(self.idx_to_filename)*len(self.augmented_type_list)*self.hidden_layer_num*self.sequence_length

    def __getitem__(self, index):
        # TODO: make it quicker !!
        augment_type = self.augmented_type_list[index//(len(self.idx_to_filename)*self.hidden_layer_num*self.sequence_length)]
        filename = self.idx_to_filename[ (index // (self.hidden_layer_num*self.sequence_length)) % len(self.idx_to_filename)].replace(".wav", "")
        layer_idx = (index // self.sequence_length) % self.hidden_layer_num
        sequence_idx = index%self.sequence_length
        
        # order: augment_type => filename => hidden_layer
        hidden_states_path = os.path.join(self.hidden_states_dir, 
                augment_type, 
                filename, 
                str(layer_idx)+".pt")

        melspetrogram_path = os.path.join(self.features_dir, 
                augment_type,
                "timbre", "melspectrogram", 
                "melspectrogram_{}.npy".format(filename))
                
        # hidden_states: (seq_len, LSTM_hidden_size)
        hidden_states = torch.load(hidden_states_path)
        # melspectrogram: (mel_banks, seq_len)
        melspectrogram = np.load(melspetrogram_path)

        # indexing timestep
        return hidden_states[sequence_idx,:], torch.tensor(melspectrogram[:, sequence_idx], dtype=torch.double)

if __name__ == "__main__":

    features_dir = "data/features/original"
    labels_dir = "data/labels"
    
    d = ReconstructionDataset("data/hidden_states", "data/features")