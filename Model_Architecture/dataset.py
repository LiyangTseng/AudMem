import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class HandCraftedDataset(Dataset):
    ''' Hand crafted audio features to predict memorability (classification) '''
    def __init__(self, config, pooling=False, mode="train"):
        super().__init__()
        self.mode = mode
        self.features_dir = config["path"]["features_dir"]
        self.labels_dir = config["path"]["labels_dir"]
        
        if self.mode != "test":
            # self.augmented_type_list = sorted(os.listdir(self.features_dir))[-1:]
            self.augmented_type_list = sorted(os.listdir(self.features_dir))[:]
        else:
            self.augmented_type_list = sorted(os.listdir(self.features_dir))[-1:]
        self.pooling = pooling

        if self.mode == "train":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[:200]
        elif self.mode == "valid":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[200:220]
        elif self.mode == "test":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[220:]
        elif self.mode == "all":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))            
        else:
            raise Exception ("Invalid dataset mode")

        self.idx_to_filename = {idx: filename for idx, filename in enumerate(self.filename_memorability_df["track"])}
        self.idx_to_mem_score = {idx: score for idx, score in enumerate(self.filename_memorability_df["score"])}

        self.features_dict = config["features"]

        self.sequential_features = []
        self.non_sequential_features = []
        self.labels = []

        for augment_type in tqdm(self.augmented_type_list):
            for audio_idx in range(len(self.idx_to_filename)):
                
                sequeutial_features_list = []
                non_sequential_features_list = []
                
                for feature_type in self.features_dict:
                    for subfeatures in self.features_dict[feature_type]:
                        feature_file_path = os.path.join(self.features_dir, augment_type, feature_type, subfeatures,
                             "{}_{}".format(subfeatures, self.idx_to_filename[audio_idx].replace("wav", "npy")))   
                        f = np.load(feature_file_path)
                        f = np.float32(f)
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
                # store features
                self.sequential_features.append(torch.from_numpy(sequential_features))
                self.non_sequential_features.append(torch.from_numpy(np.stack(non_sequential_features_list)))
                self.labels.append(torch.tensor(self.idx_to_mem_score[audio_idx]))
        

    def __len__(self):
        return len(self.idx_to_filename)*len(self.augmented_type_list)       

    def __getitem__(self, index):
        if self.mode != "test":
            return self.sequential_features[index], self.non_sequential_features[index], self.labels[index]
        else:
            return self.sequential_features[index], self.non_sequential_features[index]

class EndToEndAudioDataset(Dataset):
    ''' End-to-end audio features to predict memorability (classification) '''
    def __init__(self, config, mode="train"):
        super().__init__()
        self.mode = mode
        self.audio_dir = config["path"]["audio_dir"]
        self.labels_dir = config["path"]["labels_dir"]
        if self.mode != "test":
            self.augmented_type_list = sorted(os.listdir(self.audio_dir))[-1:]
            # self.augmented_type_list = sorted(os.listdir(self.audio_dir))[:]
        else:
            self.augmented_type_list = sorted(os.listdir(self.audio_dir))[-1:]

        if self.mode == "train":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[:200]
        elif self.mode == "valid":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[200:220]
        elif self.mode == "test":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[220:]
        elif self.mode == "all":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))            
        else:
            raise Exception ("Invalid dataset mode")

        self.idx_to_filename = {idx: filename for idx, filename in enumerate(self.filename_memorability_df["track"])}
        self.idx_to_mem_score = {idx: score for idx, score in enumerate(self.filename_memorability_df["score"])}


    def __len__(self):
        return len(self.idx_to_filename)*len(self.augmented_type_list)       

    def __getitem__(self, index):
        audio_path = os.path.join(self.audio_dir, self.augmented_type_list[index//len(self.idx_to_mem_score)], self.idx_to_filename[index%len(self.idx_to_mem_score)])
        wav, sr = torchaudio.load(audio_path)
        label = self.idx_to_mem_score[index%len(self.idx_to_mem_score)]
        
        return  wav, sr, label

class EndToEndImgDataset(Dataset):
    ''' End-to-end audio features to predict memorability (classification) '''
    def __init__(self, config, mode="train"):
        super().__init__()
        self.mode = mode
        self.img_dir = config["path"]["img_dir"]
        self.labels_dir = config["path"]["labels_dir"]
        if self.mode != "test":
            # self.augmented_type_list = sorted(os.listdir(self.img_dir))[-1:]
            self.augmented_type_list = sorted(os.listdir(self.img_dir))[:]
        else:
            self.augmented_type_list = sorted(os.listdir(self.img_dir))[-1:]

        if self.mode == "train":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[:200]
        elif self.mode == "valid":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[200:220]
        elif self.mode == "test":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[220:]
        elif self.mode == "all":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))            
        else:
            raise Exception ("Invalid dataset mode")

        self.idx_to_filename = {idx: filename for idx, filename in enumerate(self.filename_memorability_df["track"])}
        self.idx_to_mem_score = {idx: score for idx, score in enumerate(self.filename_memorability_df["score"])}
        
        self.labels = []
        self.mels_imgs = []
        self.transforms = transforms.Compose([
            transforms.Resize((config["model"]["image_size"], config["model"]["image_size"])),
            transforms.ToTensor()
        ])

        for augment_type in tqdm(self.augmented_type_list):
            for audio_idx in range(len(self.idx_to_filename)):         
                img_path = os.path.join(self.img_dir, augment_type, "mels_"+self.idx_to_filename[audio_idx].replace(".wav", ".png"))        
                image = Image.open(img_path).convert('L') # convert to grayscale
                self.mels_imgs.append(self.transforms(image))

                self.labels.append(torch.tensor(self.idx_to_mem_score[audio_idx]))


    def __len__(self):
        return len(self.idx_to_filename)*len(self.augmented_type_list)       

    def __getitem__(self, index):
        return self.mels_imgs[index], self.labels[index]

class ReconstructionDataset(Dataset):
    ''' LSTM　hidden states to reconstruct mel-spectrograms ref: https://arxiv.org/pdf/1911.01102.pdf '''
    def __init__(self, config, downsampling_factor=1, mode="train"):

        super().__init__()
        self.downsampling_factor = downsampling_factor
        self.hidden_states_dir = config["path"]["hidden_states_dir"]
        self.features_dir = config["path"]["features_dir"]
        self.hidden_layer_num = 4
        self.mode = mode
        # self.augmented_type_list = sorted(os.listdir(self.hidden_states_dir))[-1]
        if self.mode == "test":
            self.augmented_type_list = sorted(os.listdir(self.hidden_states_dir))[-1:]
        else:
            self.augmented_type_list = sorted(os.listdir(self.hidden_states_dir))[:]

        self.sequence_length = config["model"]["seq_len"]//self.downsampling_factor

        if self.mode == "train":
            filename_memorability_df = pd.read_csv("data/labels/track_memorability_scores_beta.csv")[:200]
        elif self.mode == "test":
            filename_memorability_df = pd.read_csv("data/labels/track_memorability_scores_beta.csv")[200:]
        else:
            raise Exception ("Invalid dataset mode")

            
        self.idx_to_filename = {idx: filename for idx, filename in enumerate(filename_memorability_df["track"])}

        self.hidden_states = []
        self.melspectrograms = []
        for augment_type in tqdm(self.augmented_type_list, desc="defining dataset"):
            for audio_idx in range(len(self.idx_to_filename)):
                for layer_idx in range(self.hidden_layer_num):
                    hidden_states_path = os.path.join(self.hidden_states_dir, augment_type,
                                    self.idx_to_filename[audio_idx].replace(".wav", ""), str(layer_idx)+".pt")
                    hidden_states = torch.load(hidden_states_path, map_location=torch.device('cpu'))
                    hidden_states = hidden_states[::self.downsampling_factor]               
                    self.hidden_states.append(hidden_states)
                
                if self.mode == "train":
                    melspetrogram_path = os.path.join(self.features_dir, 
                                            augment_type, "timbre", "melspectrogram", 
                                            "melspectrogram_{}.npy".format(self.idx_to_filename[audio_idx].replace(".wav", "")))

                    melspectrogram = np.load(melspetrogram_path)
                    # melspectrogram = melspectrogram[::self.downsampling_factor]
                    self.melspectrograms.append(melspectrogram)       

    def __len__(self):
        return len(self.idx_to_filename)*len(self.augmented_type_list)*self.hidden_layer_num*self.sequence_length

    def __getitem__(self, index):
        
        sequence_idx = index%self.sequence_length
        features = self.hidden_states[index//self.sequence_length][sequence_idx,:]
        features = features.detach() # this line is necessary for num_workers > 1
        if self.mode == "train":
            labels = torch.tensor(self.melspectrograms[index//(self.sequence_length*self.hidden_layer_num)][:, sequence_idx*self.downsampling_factor:(sequence_idx+1)*self.downsampling_factor], dtype=torch.double)
            return features, labels
        else:
            return features

if __name__ == "__main__":

    features_dir = "data/features/original"
    labels_dir = "data/labels"
    
    d = ReconstructionDataset("data/hidden_states", "data/features")