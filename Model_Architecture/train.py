import warnings
warnings.filterwarnings("ignore")
import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from dataset import HandCraftedDataset
from model.classification_model import MLP, LSTM

def train(args):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    print("Using device: ", device)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.model == "MLP":
        train_engineered_set = HandCraftedDataset(config=config, pooling=True, mode="train")
        valid_engineered_set = HandCraftedDataset(config=config, pooling=True, mode="valid")
        test_engineered_set = HandCraftedDataset(config=config, pooling=True, mode="test")
        
        model_config = config["model"]["MLP"]
        model = MLP(model_config=model_config).to(device).double()
    
    elif args.model == "LSTM":
        train_engineered_set = HandCraftedDataset(config=config, pooling=False, mode="train")
        valid_engineered_set = HandCraftedDataset(config=config, pooling=False, mode="valid")
        test_engineered_set = HandCraftedDataset(config=config, pooling=False, mode="test")
        
        model_config = config["model"]["LSTM"]
        model = LSTM(model_config=model_config, device=device).to(device).double()

    exp_config = config["experiment"]
    train_loader = DataLoader(dataset=train_engineered_set, batch_size=exp_config["batch_size"], shuffle=True, num_workers=15)
    valid_loader = DataLoader(dataset=valid_engineered_set, batch_size=exp_config["batch_size"], shuffle=False, num_workers=15)
    # tensorboard session
    now_tuple = time.localtime(time.time())
    date_info = '%02d-%02d-%02d_%02d:%02d'%(now_tuple[0]%100,now_tuple[1],now_tuple[2],now_tuple[3],now_tuple[4])
    writer = SummaryWriter(log_dir=os.path.join("tensorboard", args.model, date_info))

    optimizer = torch.optim.Adam(model.parameters(), exp_config["learning_rate"])

    train_loss_list = []
    valid_loss_list = []
    for epoch in range(exp_config["epochs"]):
        # train
        model.train()
        batch_train_loss = [] # record the loss of every batch
        
        for data in tqdm(train_loader, desc="train"):
            sequential_features, non_sequential_features, labels = data
            sequential_features, non_sequential_features, labels = sequential_features.to(device), non_sequential_features.to(device), labels.to(device)
 
            optimizer.zero_grad()           
            loss = model(sequential_features, non_sequential_features, labels) 
            
            batch_train_loss.append(loss.cpu().detach().numpy())
            loss.backward() # compute gradient
            optimizer.step()
            
        epoch_train_loss = np.mean(batch_train_loss)
        train_loss_list.append(epoch_train_loss)    
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        # validation
        with torch.no_grad():
            model.eval()
            data_num = 0
            batch_valid_loss = [] # record the loss of every batch
            correct_prediction = 0 # accumulate the total correct prediction in an epoch
            for data in tqdm(valid_loader, desc="valid"):
                sequential_features, non_sequential_features, labels = data
                sequential_features, non_sequential_features, labels = sequential_features.to(device), non_sequential_features.to(device), labels.to(device)
                loss = model(sequential_features, non_sequential_features, labels) 
                batch_valid_loss.append(loss.cpu().detach().numpy())
                
            epoch_valid_loss = np.mean(batch_valid_loss)
            valid_loss_list.append(epoch_valid_loss)
            writer.add_scalar("Loss/valid", epoch_valid_loss, epoch)

        print('epoch: {:2},  training loss {:.3f}  validation loss {:.3f}\n'.format(
                                    epoch+1, train_loss_list[epoch], valid_loss_list[epoch]))
        
        writer.flush()

    model_weight_dir = "weights/classification"
    model_weight_subdir = os.path.join((model_weight_dir), args.model)
    if not os.path.exists(model_weight_subdir):
        os.makedirs(model_weight_subdir)
    model_weight_path = os.path.join(model_weight_subdir, "{}.pt".format(args.model))
    torch.save(model.state_dict(), model_weight_path)
    print("weights of model '{}' saved at {}".format(args.model, model_weight_path))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train config')
    parser.add_argument("-c", "--config", help="config file location", default="config/engineered_config.yaml")
    parser.add_argument("-m", "--model", help="MLP, LSTM", default="LSTM")

    args = parser.parse_args()
    train(args)

    
 