import warnings
warnings.filterwarnings("ignore")
import os
import yaml
import argparse
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from dataset import HandCraftedDataset
from model.memorability_model import Regression_MLP, Regression_LSTM, RankNet
from utils.early_stopping_pytorch.pytorchtools import EarlyStopping

def check_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    print("[INFO]", "Using device: ", device)
    return device

def train(args):
    
    device = check_device()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    features = []
    for feature_type in config["features"]:
        features.extend(config["features"][feature_type])
    print("[INFO]", "Using features: ", features)
    
    
    if args.model == "Regression_MLP":
        train_dataset = HandCraftedDataset(config=config, pooling=True, mode="train")
        valid_dataset = HandCraftedDataset(config=config, pooling=True, mode="valid")
        
        reg_model_config = config["model"]["Regression_MLP"]
        regression_model = Regression_MLP(model_config=reg_model_config, device=device).to(device).double()
    
    elif args.model == "Regression_LSTM":
        train_dataset = HandCraftedDataset(config=config, pooling=False, mode="train")
        valid_dataset = HandCraftedDataset(config=config, pooling=False, mode="valid")
        
        reg_model_config = config["model"]["Regression_LSTM"]
        regression_model = Regression_LSTM(model_config=reg_model_config, device=device).to(device).double()
        
    else:
        raise Exception ("Invalid model")

    # tensorboard session
    now_tuple = time.localtime(time.time())
    date_info = '%02d-%02d-%02d_%02d:%02d'%(now_tuple[0]%100,now_tuple[1],now_tuple[2],now_tuple[3],now_tuple[4])
    tensorboard_dir = os.path.join("tensorboard", "train_memorability", args.model, date_info)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print("[INFO]", "tensorboard dir located at {}".format(tensorboard_dir))

    writer.add_histogram("LabeledMemScore/train", train_dataset.filename_memorability_df["score"].values)
    writer.add_histogram("LabeledMemScore/valid", valid_dataset.filename_memorability_df["score"].values)

    exp_config = config["experiment"]
    train_loader = DataLoader(dataset=train_dataset, batch_size=exp_config["batch_size"], shuffle=True, num_workers=15)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=exp_config["batch_size"], shuffle=False, num_workers=15)
    
    dataiter = iter(train_loader)
    data = dataiter.next()
    # don't know why need to wrap data in another yet, but it works
    writer.add_graph(regression_model, [data])

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=10, verbose=True)

    reg_optimizer = torch.optim.Adam(regression_model.parameters(), reg_model_config["learning_rate"])
    


    if args.model in ["Regression_MLP", "Regression_LSTM"]:
        
        smallest_valid_loss = 10000
        for epoch in range(exp_config["epochs"]):
            # train
            print("\nepoch: {}/{}".format(epoch+1, exp_config["epochs"]))

            regression_model.train()
            train_reg_loss, train_rank_loss, train_total_loss = [], [], [] # record the loss of every batch
            train_reg_prediction, train_rank_prediction = [], []
            for data in tqdm(train_loader, desc="train"):
                
                reg_optimizer.zero_grad()

                predicted_mem_score, labeled_mem_score = regression_model(data) 
                train_reg_prediction.append(predicted_mem_score)
                reg_loss = nn.MSELoss()(predicted_mem_score, labeled_mem_score.reshape(-1,1))
                train_reg_loss.append(reg_loss.cpu().detach().numpy())
                
                train_rank_prediction.append(predicted_mem_score[:data[0].size(0)//2] > predicted_mem_score[data[0].size(0)//2:])
                predicted_binary_rank = nn.Sigmoid()(predicted_mem_score[:data[0].size(0)//2] - predicted_mem_score[data[0].size(0)//2:])
                labeled_binary_rank = torch.unsqueeze((labeled_mem_score[:data[0].size(0)//2]>labeled_mem_score[data[0].size(0)//2:]).double(), 1).to(device)
                rank_loss = nn.BCELoss()(predicted_binary_rank, labeled_binary_rank)
                train_rank_loss.append(rank_loss.cpu().detach().numpy())

                loss = reg_loss + 0.2*rank_loss
                train_total_loss.append(loss.cpu().detach().numpy())

                loss.backward()
                reg_optimizer.step()

            epoch_train_total_loss = np.mean(train_total_loss)
            epoch_train_reg_loss = np.mean(train_reg_loss)
            epoch_train_rank_loss = np.mean(train_rank_loss)

            writer.add_histogram("PredictedMemScore/train", torch.cat(train_reg_prediction), epoch+1)    
            writer.add_scalar("RegLoss/train", epoch_train_reg_loss, epoch)
            writer.add_scalar("RankLoss/train", epoch_train_rank_loss, epoch)
            writer.add_scalar("TotalLoss/train", epoch_train_total_loss, epoch)
            
            print("train total loss {:.3f},  train reg loss {:.3f}, train rank loss {:.3f}".format(
                                        epoch_train_total_loss, epoch_train_reg_loss, epoch_train_rank_loss))

            # validation
            with torch.no_grad():
                regression_model.eval()
                valid_reg_loss, valid_rank_loss, valid_total_loss = [], [], [] # record the loss of every batch
                valid_reg_prediction, valid_rank_prediction = [], []
                for data in tqdm(valid_loader, desc="valid"):

                    predicted_mem_score, labeled_mem_score = regression_model(data) 
                    valid_reg_prediction.append(predicted_mem_score)
                    reg_loss = nn.MSELoss()(predicted_mem_score, labeled_mem_score.reshape(-1,1))
                    valid_reg_loss.append(reg_loss.cpu().detach().numpy())
                    
                    valid_rank_prediction.append(predicted_mem_score[:data[0].size(0)//2] > predicted_mem_score[data[0].size(0)//2:])
                    predicted_binary_rank = nn.Sigmoid()(predicted_mem_score[:data[0].size(0)//2] - predicted_mem_score[data[0].size(0)//2:])
                    labeled_binary_rank = torch.unsqueeze((labeled_mem_score[:data[0].size(0)//2]>labeled_mem_score[data[0].size(0)//2:]).double(), 1).to(device)
                    rank_loss = nn.BCELoss()(predicted_binary_rank, labeled_binary_rank)
                    valid_rank_loss.append(rank_loss.cpu().detach().numpy())

                    loss = reg_loss + 0.2*rank_loss
                    valid_total_loss.append(loss.cpu().detach().numpy())
                    
                epoch_valid_total_loss = np.mean(valid_total_loss)
                epoch_valid_reg_loss = np.mean(valid_reg_loss)
                epoch_valid_rank_loss = np.mean(valid_rank_loss)
                writer.add_histogram("PredictedMemScore/valid", torch.cat(valid_reg_prediction), epoch+1)    
                writer.add_scalar("RegLoss/valid", epoch_valid_reg_loss, epoch)
                writer.add_scalar("RankLoss/valid", epoch_valid_rank_loss, epoch)
                writer.add_scalar("TotalLoss/valid", epoch_valid_total_loss, epoch)

                if epoch_valid_total_loss < smallest_valid_loss:
                    best_model = regression_model
                    
            for name, param in regression_model.named_parameters():
                writer.add_histogram(name, param, epoch+1)
            
            print("valid total loss {:.3f},  valid reg loss {:.3f}, valid rank loss {:.3f}".format(
                                        epoch_valid_total_loss, epoch_valid_reg_loss, epoch_valid_rank_loss))
            
            writer.flush()
            early_stopping(epoch_valid_total_loss, regression_model)
            if early_stopping.early_stop:
                print("Early Stoppoing")
                break
    else:
        raise Exception ("Invalid model")


    model_weight_dir = "weights/train_memorability"
    model_weight_subdir = os.path.join((model_weight_dir), args.model, date_info)
    if not os.path.exists(model_weight_subdir):
        os.makedirs(model_weight_subdir)
    model_weight_path = os.path.join(model_weight_subdir, "{}.pt".format(args.model))
    torch.save(best_model.state_dict(), model_weight_path)
    print("[INFO]", "weights of best model '{}' saved at {}".format(args.model, model_weight_path))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train config')
    parser.add_argument("-c", "--config", help="config file location", default="config/engineered_config.yaml")
    parser.add_argument("-m", "--model", help="Regression_MLP, Regression_LSTM", default="Regression_MLP")
    parser.add_argument("-e", "--early_stopping", help="early stoppping or not", default=True, type=bool)

    args = parser.parse_args()
    train(args)

    
 