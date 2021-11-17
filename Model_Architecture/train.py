import warnings
warnings.filterwarnings("ignore")
import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    train_loader = DataLoader(dataset=train_engineered_set, batch_size=exp_config["batch_size"], shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_engineered_set, batch_size=exp_config["batch_size"], shuffle=False, num_workers=2)
    
    optimizer = torch.optim.Adam(model.parameters(), exp_config["learning_rate"])


    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    for epoch in range(exp_config["epochs"]):
        # train
        model.train()
        data_num = 0
        batch_train_loss = [] # record the loss of every batch
        correct_prediction = 0 # accumulate the total correct prediction in an epoch
        
        for data in tqdm(train_loader, desc="train"):
            sequential_features, non_sequential_features, labels = data
            sequential_features, non_sequential_features, labels = sequential_features.to(device), non_sequential_features.to(device), labels.to(device)
 
            optimizer.zero_grad()           
            batch_size, correct_cnt, loss = model(sequential_features, non_sequential_features, labels) 
        
            data_num += batch_size
            batch_train_loss.append(loss.cpu().detach().numpy())
            correct_prediction += correct_cnt
            loss.backward() # compute gradient
            optimizer.step()
        
        train_loss_list.append(np.mean(batch_train_loss))
        train_acc_list.append(correct_prediction/data_num)
        
        # validation
        with torch.no_grad():
            model.eval()
            data_num = 0
            batch_valid_loss = [] # record the loss of every batch
            correct_prediction = 0 # accumulate the total correct prediction in an epoch
            for data in tqdm(valid_loader, desc="valid"):
                sequential_features, non_sequential_features, labels = data
                sequential_features, non_sequential_features, labels = sequential_features.to(device), non_sequential_features.to(device), labels.to(device)
                batch_size, correct_cnt, loss = model(sequential_features, non_sequential_features, labels) 
                data_num += batch_size
                batch_valid_loss.append(loss.cpu().detach().numpy())
                correct_prediction += correct_cnt
                
            valid_loss_list.append(np.mean(batch_valid_loss))
            valid_acc_list.append(correct_prediction/data_num)
        
        print('epoch: {:2},  training loss {:.2f}  training acc: {:.2f}  validation loss {:.2f}  validation acc: {:.2f}\n'.format(
            epoch+1, train_loss_list[epoch], train_acc_list[epoch], valid_loss_list[epoch], valid_acc_list[epoch]))
        
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

    
 