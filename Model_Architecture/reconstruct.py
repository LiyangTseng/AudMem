import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import torch
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from models.probing_model import Probing_Model
from models.memorability_model import Regression_MLP, Regression_LSTM
from src.dataset  import HandCraftedDataset, ReconstructionDataset
# https://github.com/Bjarten/early-stopping-pytorch
from utils.early_stopping_pytorch.pytorchtools import EarlyStopping

def check_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    print("[INFO]", "using device: ", device)
    return device

def get_hidden_states(args, recon_config):
    " use trained LSTM weights to get all hidden states " 

    device = check_device()

    print("\n[INFO]", "getting hidden states\n")

    with open("config/engineered_config.yaml", 'r') as f:
        memo_config = yaml.load(f, Loader=yaml.FullLoader)
    # memorability_model_config = memorability_config["model"]["Regression_LSTM"]
    memo_model_config = memo_config["model"][args.memorability_model]

    if args.memorability_model == "Regression_MLP":
        # ==== same as train.py ====
        all_engineered_set = HandCraftedDataset(config=memo_config, pooling=True, split="all") 
        sequential_input_size, non_sequential_input_size = memo_model_config["sequential_input_size"], memo_model_config["non_sequential_input_size"] 
        model = Regression_MLP(model_config=memo_model_config, device=device).to(device).double()
        model.load_state_dict(torch.load("weights/train_memorability/Regression_MLP/Regression_MLP.pt"))
        # TODO: finish the remaining part
        
    elif args.memorability_model == "Regression_LSTM":
        all_engineered_set = HandCraftedDataset(config=memo_config, pooling=False, split="all") 
        sequential_input_size, non_sequential_input_size = memo_model_config["sequential_input_size"], memo_model_config["non_sequential_input_size"] 
        model = Regression_LSTM(model_config=memo_model_config, device=device).to(device).double()
        model.load_state_dict(torch.load("weights/train_memorability/Regression_LSTM/Regression_LSTM.pt"))

        model.eval()
        for i in tqdm(range(len(all_engineered_set))):

            hidden_states_subdir = os.path.join(recon_config["path"]["hidden_states_dir"], all_engineered_set.augmented_type_list[i//len(all_engineered_set.idx_to_filename)],
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
        print("[INFO]", "hidden states saved at {}".format(recon_config["path"]["hidden_states_dir"]))

    else:
        raise Exception("no implement error")
    # ========================== 
    

def train_valid_split(config, dataset, batch_size):

    ''' ref: https://stackoverflow.com/a/50544887/4440387 '''
    
    validation_split = 0.2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(train_indices)
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SequentialSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler, num_workers=10)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=10)
    return train_loader, valid_loader

def train_reconstruction(args, config):

    print("\n[INFO]", "start training audio reconstruction\n")
    device = check_device()
    model_config = config["model"]

    train_dataset = ReconstructionDataset(config, downsampling_factor=model_config["downsampling_factor"], mode="train")
    
    model = Probing_Model(model_config=model_config, mode="train").to(device).double()
    
    now_tuple = time.localtime(time.time())
    date_info = '%02d-%02d-%02d_%02d:%02d'%(now_tuple[0]%100,now_tuple[1],now_tuple[2],now_tuple[3],now_tuple[4])
    tensorboard_dir = os.path.join("tensorboard", "reconstruct_audios", args.probing_model, date_info)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print("[INFO]", "tensorboard dir located at {}".format(tensorboard_dir))

    train_loader, valid_loader = train_valid_split(config=config, dataset=train_dataset, batch_size=config["model"]["seq_len"]//config["model"]["downsampling_factor"])
    
    dataiter = iter(train_loader)
    data = dataiter.next()
    data = [datum.to(device) for datum in data]
    # don't know why need to wrap data in another yet, but it works
    writer.add_graph(model, data)
    

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=5, verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), config["experiment"]["learning_rate"])
    
    train_loss_list = []
    valid_loss_list = []

    smallest_valid_loss = 10000
    for epoch in range(config["experiment"]["epochs"]):
        model.train()
        batch_train_loss = [] # record the loss of every batchd
        
        for data in tqdm(train_loader, desc="train"):
            hidden_states, melspectrogram = data
            hidden_states, melspectrogram = hidden_states.to(device), melspectrogram.to(device)
            
            optimizer.zero_grad()
            
            _, loss = model(hidden_states, melspectrogram)
            
            batch_train_loss.append(loss.cpu().detach().numpy())
            loss.backward() # compute gradient
            optimizer.step()
        
        epoch_train_loss = np.mean(batch_train_loss)
        train_loss_list.append(epoch_train_loss)    
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)

        with torch.no_grad():
            model.eval()
            batch_valid_loss = [] # record the loss of every batch
            for idx, data in enumerate(tqdm(valid_loader, desc="valid")):
                hidden_states, melspectrogram = data
                hidden_states, melspectrogram = hidden_states.to(device), melspectrogram.to(device)
                
                predicted_mels, loss = model(hidden_states, melspectrogram)
                batch_valid_loss.append(loss.cpu().detach().numpy())
                
                if idx%40 == 0:
                    # (N, n_mels, downsmapling_factor) 
                    melspectrogram = melspectrogram.permute(1,0,2)
                    # (n_mels, N, downsmapling_factor) 
                    melspectrogram = melspectrogram.reshape(-1, melspectrogram.size(1)*melspectrogram.size(2))
                    # (n_mels, N*downsmapling_factor) 
                    melspectrogram = melspectrogram.cpu().detach().numpy()
                    fig = convert_mels_to_fig(melspectrogram)
                    writer.add_figure("{}/valid_label".format(idx), fig, epoch)

                    # (N, n_mels, downsmapling_factor) 
                    predicted_mels = predicted_mels.permute(1,0,2)
                    # (n_mels, N, downsmapling_factor) 
                    predicted_mels = predicted_mels.reshape(-1, predicted_mels.size(1)*predicted_mels.size(2))
                    # (n_mels, N*downsmapling_factor) 
                    predicted_mels = predicted_mels.cpu().detach().numpy()
                    fig = convert_mels_to_fig(predicted_mels)
                    writer.add_figure("{}/valid_pred".format(idx), fig, epoch)
                    
                
            epoch_valid_loss = np.mean(batch_valid_loss)
            valid_loss_list.append(epoch_valid_loss)
            writer.add_scalar("Loss/valid", epoch_valid_loss, epoch)
            if epoch_valid_loss < smallest_valid_loss:
                best_model = model
        
        print('epoch: {:2},  training loss {:.4f},  validation loss {:.4f}\n'.format(
            epoch+1, train_loss_list[epoch], valid_loss_list[epoch]))
        writer.flush()

        early_stopping(epoch_valid_loss, model)
        if early_stopping.early_stop:
            print("Early Stoppoing")
            break

        
    model_weight_dir = "weights/reconstruct_audios"
    model_weight_subdir = os.path.join((model_weight_dir), args.probing_model)
    if not os.path.exists(model_weight_subdir):
        os.makedirs(model_weight_subdir)
    model_weight_path = os.path.join(model_weight_subdir, "{}.pt".format(args.probing_model))
    torch.save(best_model.state_dict(), model_weight_path)
    print("[INFO]", "weights of best model '{}' saved at {}".format(args.probing_model, model_weight_path))

def convert_mels_to_fig(predicted_mels):
    ''' return figure (from matplotlib) of melspectrogram '''
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set(title="Log-Power spectrogram")
    S_dB = librosa.power_to_db(predicted_mels, ref=np.max)
    p = librosa.display.specshow(S_dB, ax=ax, y_axis='log', x_axis='time')
    fig.colorbar(p, ax=ax, format="%+2.0f dB")

    return fig

def inference_reconstruction(args, config):
    
    print("\n[INFO]", "start testing audio reconstruction\n")
    device = check_device()
    test_dataset = ReconstructionDataset(config, downsampling_factor=config["model"]["downsampling_factor"], mode="test")
    
    # set size of one batch to entire sequence length
    test_loader = DataLoader(dataset=test_dataset, batch_size=config["model"]["seq_len"]//config["model"]["downsampling_factor"], shuffle=False, num_workers=15)


    model_config = config["model"]
    model = Probing_Model(model_config=model_config, mode="test").to(device).double()
    model_weight_dir = "weights/reconstruct_audios"
    model_weight_path = os.path.join((model_weight_dir), args.probing_model, "{}.pt".format(args.probing_model))
    model.load_state_dict(torch.load(model_weight_path))

    reconstructed_audio_dir = "data/reconstructed_audios"
    reconstructed_mels_dir = "data/reconstructed_mels"
    for idx, data in enumerate(tqdm(test_loader, desc="inference")):
        hidden_states = data
        hidden_states = hidden_states.to(device)
        
        predicted_mels = model(hidden_states)
        # (N, n_mels, downsmapling_factor) 

        predicted_mels = predicted_mels.permute(1,0,2)
        # (n_mels, N, downsmapling_factor) 

        predicted_mels = predicted_mels.reshape(-1, predicted_mels.size(1)*predicted_mels.size(2))
        # (n_mels, N*downsmapling_factor) 

        predicted_mels = predicted_mels.cpu().detach().numpy()

        augment_type_idx = idx//(len(test_dataset.idx_to_filename)*test_dataset.hidden_layer_num)
        audio_idx = (idx%((len(test_dataset.idx_to_filename)*test_dataset.hidden_layer_num)))//test_dataset.hidden_layer_num
        layer_idx = idx%test_dataset.hidden_layer_num

        if args.save_mels:
            reconstructed_mels_subdir = os.path.join(reconstructed_mels_dir, test_dataset.augmented_type_list[augment_type_idx], test_dataset.idx_to_filename[audio_idx].replace(".wav", ""))
            if not os.path.exists(reconstructed_mels_subdir):
                os.makedirs(reconstructed_mels_subdir)
            reconstructed_mels_path = os.path.join(reconstructed_mels_subdir, "{}.jpg".format(layer_idx))
            fig = convert_mels_to_fig(predicted_mels)
            fig.savefig(reconstructed_mels_path)

        reconstructed_audio_subdir = os.path.join(reconstructed_audio_dir, test_dataset.augmented_type_list[augment_type_idx], test_dataset.idx_to_filename[audio_idx].replace(".wav", ""))
        if not os.path.exists(reconstructed_audio_subdir):
            os.makedirs(reconstructed_audio_subdir)
        reconstructed_audio_path = os.path.join(reconstructed_audio_subdir, "{}.wav".format(layer_idx))
    
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(predicted_mels)
        sf.write(reconstructed_audio_path, reconstructed_audio, samplerate=22050)

    print("[INFO]", "reconstructed audios saved at {}".format(reconstructed_audio_dir))
    print("[INFO]", "reconstructed melspectrograms saved at {}".format(reconstructed_mels_dir))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='reconstruct audio')
    parser.add_argument("-c", "--config", help="config file location", default="config/reconstruct_config.yaml")
    parser.add_argument("-g", "--hidden_states", help="get hidden states or not", default=True, type=bool)
    parser.add_argument("-t", "--train", help="train or not", default=True, type=bool)
    parser.add_argument("-i", "--inference", help="test or not", default=True, type=bool)
    parser.add_argument("-m", "--memorability_model", help="Regression_MLP, Regression_LSTM, Ranking_LSTM, Ranking_MLP", default="Regression_LSTM")
    parser.add_argument("-p", "--probing_model", help="highway", default="highway")
    parser.add_argument("-e", "--early_stopping", help="early stoppping or not", default=True, type=bool)
    parser.add_argument("-s", "--save_mels", help="save predicted mel-spectrograms or not", default=True, type=bool)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.hidden_states:
        get_hidden_states(args, config)
    if args.train:
        train_reconstruction(args, config)
    if args.inference:
        inference_reconstruction(args, config)
        