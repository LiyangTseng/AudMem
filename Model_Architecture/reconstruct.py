import os
import numpy as np
import librosa
import soundfile as sf
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
from model.probing_model import Probing_Model
from model.classification_model import LSTM
from dataset import HandCraftedDataset, ReconstructionDataset


def check_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    print("Using device: ", device)
    return device

def get_hidden_states(config):
    " use trained LSTM weights to get all hidden states " 

    device = check_device()
    # ==== same as train.py ====
    all_engineered_set = HandCraftedDataset(config=config, pooling=False, mode="all") 
    sequential_input_size, non_sequential_input_size = all_engineered_set[0][0].size(1), all_engineered_set[0][1].size(0)
    model = LSTM(sequential_input_size, non_sequential_input_size, hidden_size=320, num_layers=4, device=device, bidirectional=True).to(device).double()
    # ========================== 
    model.load_state_dict(torch.load("weights/LSTM/LSTM.pt"))
    model.eval()

    for i in tqdm(range(len(all_engineered_set))):

        hidden_states_subdir = os.path.join(config["path"]["hidden_states_dir"], all_engineered_set.augmented_type_list[i//len(all_engineered_set.idx_to_filename)],
                                 all_engineered_set.idx_to_filename[i%len(all_engineered_set.idx_to_filename)].replace(".wav", ""))
        if not os.path.exists(hidden_states_subdir):
            os.makedirs(hidden_states_subdir)

        sequential_features, _, _ = all_engineered_set[i]
        # unsqueeze to create batch size = 1, serve as model input (1, seq_len, feature_size)
        sequential_features = torch.unsqueeze(sequential_features, 0)
        sequential_features = sequential_features.to(device)
        for idx, layer in enumerate(model.LSTM):
            inputs = sequential_features if layer == model.LSTM[0] else hidden_states
            hidden_states, _ = layer(inputs)
            # save squeezed tensor (seq_len, feature_size) 
            torch.save(torch.squeeze(hidden_states, 0), os.path.join(hidden_states_subdir, str(idx)+".pt"))

def train_valid_split(config, dataset):

    ''' ref: https://stackoverflow.com/a/50544887/4440387 '''
    
    batch_size = config["experiment"]["batch_size"]
    validation_split = 0.2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler, num_workers=10)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=10)
    return train_loader, valid_loader

def train_reconstruction(args, config):

    device = check_device()
    train_dataset = ReconstructionDataset(config)
    train_loader, valid_loader = train_valid_split(config=config, dataset=train_dataset)
    
    model_config = config["model"]
    model = Probing_Model(model_config=model_config, mode="train").to(device).double()
    
    optimizer = torch.optim.Adam(model.parameters(), config["experiment"]["learning_rate"])
    
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(config["experiment"]["epochs"]):
        model.train()
        batch_train_loss = [] # record the loss of every batch
        
        for data in tqdm(train_loader, desc="train"):
            hidden_states, melspectrogram = data
            hidden_states, melspectrogram = hidden_states.to(device), melspectrogram.to(device)
            
            optimizer.zero_grad()
            
            _, loss = model(hidden_states, melspectrogram)
            
            batch_train_loss.append(loss.cpu().detach().numpy())
            loss.backward() # compute gradient
            optimizer.step()
        
        train_loss_list.append(np.mean(batch_train_loss))

        with torch.no_grad():
            model.eval()
            batch_valid_loss = [] # record the loss of every batch
            for data in tqdm(valid_loader, desc="valid"):
                hidden_states, melspectrogram = data
                hidden_states, melspectrogram = hidden_states.to(device), melspectrogram.to(device)
                
                _, loss = model(hidden_states, melspectrogram)
                
                batch_valid_loss.append(loss.cpu().detach().numpy())
                
            valid_loss_list.append(np.mean(batch_valid_loss))
        
        print('epoch: {:2},  training loss {:.2f},  validation loss {:.2f}\n'.format(
            epoch+1, train_loss_list[epoch], valid_loss_list[epoch]))
        
    model_weight_dir = "weights/probing"
    model_weight_subdir = os.path.join((model_weight_dir), args.model)
    if not os.path.exists(model_weight_subdir):
        os.makedirs(model_weight_subdir)
    model_weight_path = os.path.join(model_weight_subdir, "{}.pt".format(args.model))
    torch.save(model.state_dict(), model_weight_path)
    print("weights of model '{}' saved at {}".format(args.model, model_weight_path))

def inference_reconstruction(args, config):
    
    device = check_device()
    test_dataset = ReconstructionDataset(config, mode="test")
    
    # set size of one batch to entire sequence length
    test_loader = DataLoader(dataset=test_dataset, batch_size=config["model"]["seq_len"], shuffle=False, num_workers=10)


    model_config = config["model"]
    model = Probing_Model(model_config=model_config, mode="test").to(device).double()
    model_weight_dir = "weights/probing"
    model_weight_path = os.path.join((model_weight_dir), args.model, "{}.pt".format(args.model))
    model.load_state_dict(torch.load(model_weight_path))

    reconstruct_audio_dir = "data/reconstructed_audios"
    for idx, data in enumerate(tqdm(test_loader, desc="inference")):
        hidden_states = data
        hidden_states = hidden_states.to(device)
        
        predicted_mels = model(hidden_states)
        predicted_mels = predicted_mels.cpu().detach().numpy().T
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(predicted_mels)


        augment_type_idx = idx//(len(test_dataset.idx_to_filename)*test_dataset.hidden_layer_num)
        audio_idx = (idx%((len(test_dataset.idx_to_filename)*test_dataset.hidden_layer_num)))//test_dataset.hidden_layer_num
        layer_idx = idx%test_dataset.hidden_layer_num

        
        reconstruct_audio_subdir = os.path.join(reconstruct_audio_dir, test_dataset.augmented_type_list[augment_type_idx], test_dataset.idx_to_filename[audio_idx].replace(".wav", ""))
        if not os.path.exists(reconstruct_audio_subdir):
            os.makedirs(reconstruct_audio_subdir)
        reconstruct_audio_path = os.path.join(reconstruct_audio_subdir, "{}.wav".format(layer_idx))
        
        sf.write(reconstruct_audio_path, reconstructed_audio, samplerate=22050)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='reconstruct audio')
    parser.add_argument("-c", "--config", help="config file location", default="config/reconstruct_config.yaml")
    parser.add_argument("-g", "--hidden_states", help="get hidden states or not", default=False, type=bool)
    parser.add_argument("-t", "--train", help="train or not", default=True, type=bool)
    parser.add_argument("-i", "--inference", help="test or not", default=True, type=bool)
    parser.add_argument("-m", "--model", help="highway", default="highway")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("config file loaded")

    if args.hidden_states:
        get_hidden_states(config)
    if args.train:
        train_reconstruction(args, config)
    if args.inference:
        inference_reconstruction(args, config)
    