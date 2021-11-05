import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class HandCraftedDataset(Dataset):
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

        self.features_dict = {"chords": ["chroma"], "emotions": ["static_arousal", "static_valence"],
                                "rhythms": ["tempogram"], "timbre": ["mfcc"] }
        
    def __len__(self):
        return len(self.idx_to_filename)       

    def __getitem__(self, index):
        
        # TODO: how to use sequential features? now using tempogral pooling
        concate_features = np.array([])
        for feature_type in self.features_dict:
            for subfeatures in self.features_dict[feature_type]:
                f = np.load(os.path.join(self.features_dir, feature_type, subfeatures, "{}_{}".format(subfeatures, self.idx_to_filename[index].replace("wav", "npy"))))
                if feature_type == "emotions":
                    # features not sequential
                    concate_features = np.concatenate([concate_features, f.reshape(-1)])
                else:
                    # features are sequential => temporal mean pooling
                    concate_features = np.concatenate([concate_features, f.mean(axis=1)])
        
        labels = self.idx_to_mem_score[index]

        return torch.tensor(concate_features), torch.tensor(labels, dtype=torch.double)
        
        
if __name__ == "__main__":

    features_dir = "../Feature_Extraction/features"
    labels_dir = "../Feature_Extraction/labels"
    
    train_engineered_set = HandCraftedDataset(features_dir=features_dir, labels_dir=labels_dir, mode="train")
    valid_engineered_set = HandCraftedDataset(features_dir=features_dir, labels_dir=labels_dir, mode="valid")
    test_engineered_set = HandCraftedDataset(features_dir=features_dir, labels_dir=labels_dir, mode="test")
    
    train_loader = DataLoader(dataset=train_engineered_set, batch_size=5, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_engineered_set, batch_size=5, shuffle=False, num_workers=2)

    for _, data in enumerate(train_loader):
        features, labels = data
        print(features.size(), labels.size())
        break