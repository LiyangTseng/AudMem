import warnings
warnings.filterwarnings("ignore")
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import HandCraftedDataset
from model.engineered_model import MLP

def train(args):
    
    with open(args.config, 'r') as f:
        hyper_parameters = yaml.load(f, Loader=yaml.FullLoader)
    
    train_engineered_set = HandCraftedDataset(features_dir=args.feature_dir, labels_dir=args.label_dir, mode="train")
    valid_engineered_set = HandCraftedDataset(features_dir=args.feature_dir, labels_dir=args.label_dir, mode="valid")
    test_engineered_set = HandCraftedDataset(features_dir=args.feature_dir, labels_dir=args.label_dir, mode="test")
    
    train_loader = DataLoader(dataset=train_engineered_set, batch_size=hyper_parameters["experiment"]["batch_size"], shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_engineered_set, batch_size=hyper_parameters["experiment"]["batch_size"], shuffle=False, num_workers=2)

    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu") 
    device = "cpu"
    print("Using device: ", device)

    if args.model == "MLP":
        model = MLP().to(device).double()
    elif args.model == "LSTM":
        pass
    
    optimizer = torch.optim.Adam(model.parameters(), hyper_parameters["experiment"]["learning_rate"])


    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    for epoch in range(hyper_parameters["experiment"]["epochs"]):
        # train
        model.train()
        data_num = 0
        batch_train_loss = [] # record the loss of every batch
        correct_prediction = 0 # accumulate the total correct prediction in an epoch
        for _, data in enumerate(train_loader):
            features, labels = data
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad() 
            batch_size, correct_cnt, loss = model(features, labels) 
            data_num += batch_size
            batch_train_loss.append(loss.detach().numpy())
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
            for _, data in enumerate(valid_loader):
                features, labels = data
                features, labels = features.to(device), labels.to(device)
                batch_size, correct_cnt, loss = model(features, labels) 
                data_num += batch_size
                batch_valid_loss.append(loss.detach().numpy())
                correct_prediction += correct_cnt
                
            valid_acc_list.append(np.mean(batch_valid_loss))
            valid_loss_list.append(correct_prediction/data_num)
        
        print('epoch: {:2},  training loss {:.2f}  training acc: {:.2f}  validation loss {:.2f}  validation acc: {:.2f}'.format(
            epoch+1, train_loss_list[epoch], train_acc_list[epoch], valid_loss_list[epoch], valid_acc_list[epoch]))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train config')
    parser.add_argument("-p", "--preprocess", help="preprocessing or not", default=False, type=bool)
    parser.add_argument("-c", "--config", help="config file location", default="config/engineered_config.yaml")
    parser.add_argument("-f", "--feature_dir", help="directory of features", default="../Feature_Extraction/features")
    parser.add_argument("-l", "--label_dir", help="directory of labels", default="../Feature_Extraction/labels")
    parser.add_argument("-m", "--model", help="MLP, LSTM", default="MLP")

    args = parser.parse_args()
    train(args)

    
 