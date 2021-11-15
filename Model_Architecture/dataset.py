import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class HandCraftedDataset_Pooling(Dataset):
    def __init__(self, features_dir, labels_dir, mode="train"):
        super().__init__()
        self.features_dir = features_dir
        self.lables_dir = labels_dir

        if mode == "train":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.lables_dir, "track_memorability_scores_beta.csv"))[:200]
        elif mode == "valid":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.lables_dir, "track_memorability_scores_beta.csv"))[200:220]
        elif mode == "test":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.lables_dir, "track_memorability_scores_beta.csv"))[220:]
        else:
            raise Exception ("Invalid dataset mode")

        self.idx_to_filename = {idx: filename for idx, filename in enumerate(self.filename_memorability_df["track"])}
        self.idx_to_mem_score = {idx: score for idx, score in enumerate(self.filename_memorability_df["score"])}

        self.features_dict = {"chords": ["chroma"], "rhythms": ["tempogram"], "timbre": ["mfcc"], "emotions": ["static_arousal", "static_valence"]}
        
    def __len__(self):
        return len(self.idx_to_filename)       

    def __getitem__(self, index):
        
        # TODO: how to use sequential features in MLP? now using tempogral pooling to compress sequential features
        sequeutial_features_list = []
        non_sequential_features_list = []
        
        for feature_type in self.features_dict:
            for subfeatures in self.features_dict[feature_type]:
                feature_file_path = os.path.join(self.features_dir, feature_type, subfeatures, "{}_{}".format(subfeatures, self.idx_to_filename[index].replace("wav", "npy")))
                
                f = np.load(feature_file_path)
                if feature_type == "emotions":
                    # features not sequential
                    # concate_features = np.concatenate([concate_features, f.reshape(-1)])
                    non_sequential_features_list.append(f)
                else:
                    # features are sequential => temporal mean pooling
                    # concate_features = np.concatenate([concate_features, f.mean(axis=1)])
                    sequeutial_features_list.append(f.mean(axis=1))
        
        sequential_features = np.concatenate(sequeutial_features_list, axis=0)
        non_sequential_features = np.array(non_sequential_features_list)
        labels = self.idx_to_mem_score[index]
    
        return torch.tensor(sequential_features), torch.tensor(non_sequential_features), torch.tensor(labels, dtype=torch.double)
        
class HandCraftedDataset_Sequential(Dataset):
    def __init__(self, features_dir, labels_dir, mode="train"):
        super().__init__()
        self.features_dir = features_dir
        self.lables_dir = labels_dir

        if mode == "train":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.lables_dir, "track_memorability_scores_beta.csv"))[:200]
        elif mode == "valid":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.lables_dir, "track_memorability_scores_beta.csv"))[200:220]
        elif mode == "test":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.lables_dir, "track_memorability_scores_beta.csv"))[220:]
        else:
            raise Exception ("Invalid dataset mode")

        self.idx_to_filename = {idx: filename for idx, filename in enumerate(self.filename_memorability_df["track"])}
        self.idx_to_mem_score = {idx: score for idx, score in enumerate(self.filename_memorability_df["score"])}

        self.features_dict = {"chords": ["chroma"], "rhythms": ["tempogram"], "timbre": ["mfcc"], "emotions": ["static_arousal", "static_valence"]}
        
    def __len__(self):
        return len(self.idx_to_filename)       

    def __getitem__(self, index):

        # TODO: how to use non-sequential features in RNN? now concate hidden states with 
        sequeutial_features_list = []
        non_sequential_features_list = []
        for feature_type in self.features_dict:
            for subfeatures in self.features_dict[feature_type]:
                feature_file_path = os.path.join(self.features_dir, feature_type, subfeatures, "{}_{}".format(subfeatures, self.idx_to_filename[index].replace("wav", "npy")))
                
                f = np.load(feature_file_path)
                if feature_type == "emotions":
                    # features not sequential
                    non_sequential_features_list.append(f)
                else:
                    sequeutial_features_list.append(f)
        
        sequential_features = np.concatenate(sequeutial_features_list, axis=0)
        # reshape to (seq_length, input_size)
        sequential_features = np.transpose(sequential_features)

        non_sequential_features = np.array(non_sequential_features_list)
        labels = self.idx_to_mem_score[index]

        return torch.tensor(sequential_features), torch.tensor(non_sequential_features), torch.tensor(labels, dtype=torch.double)
        
if __name__ == "__main__":

    features_dir = "data/features/original"
    labels_dir = "data/labels"
    
    train_engineered_set = HandCraftedDataset_Pooling(features_dir=features_dir, labels_dir=labels_dir, mode="train")
    valid_engineered_set = HandCraftedDataset_Pooling(features_dir=features_dir, labels_dir=labels_dir, mode="valid")
    test_engineered_set = HandCraftedDataset_Pooling(features_dir=features_dir, labels_dir=labels_dir, mode="test")
    # train_engineered_set = HandCraftedDataset_Sequential(features_dir=features_dir, labels_dir=labels_dir, mode="train")
    # valid_engineered_set = HandCraftedDataset_Sequential(features_dir=features_dir, labels_dir=labels_dir, mode="valid")
    # test_engineered_set = HandCraftedDataset_Sequential(features_dir=features_dir, labels_dir=labels_dir, mode="test")
    
    train_loader = DataLoader(dataset=train_engineered_set, batch_size=1, shuffle=False, num_workers=1)
    valid_loader = DataLoader(dataset=valid_engineered_set, batch_size=1, shuffle=False, num_workers=1)

    for _, data in enumerate(train_loader):
        sequential_features, non_sequential_features, labels = data
        print(sequential_features.size(), non_sequential_features.size(), labels.size())
        break