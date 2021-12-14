import warnings
warnings.filterwarnings("ignore")
import os
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from dataset import HandCraftedDataset
from model.memorability_model import Regression_MLP, Regression_LSTM, RankNet
from utils.early_stopping_pytorch.pytorchtools import EarlyStopping

def train(args):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    print("Using device: ", device)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    features = []
    for feature_type in config["features"]:
        features.extend(config["features"][feature_type])
    print("Using features: ", features)
    
    
    if args.model == "Regression_MLP":
        train_dataset = HandCraftedDataset(config=config, pooling=True, mode="train")
        valid_dataset = HandCraftedDataset(config=config, pooling=True, mode="valid")
        
        model_config = config["model"]["Regression_MLP"]
        model = Regression_MLP(model_config=model_config, device=device).to(device).double()
    
    elif args.model == "Regression_LSTM":
        train_dataset = HandCraftedDataset(config=config, pooling=False, mode="train")
        valid_dataset = HandCraftedDataset(config=config, pooling=False, mode="valid")
        
        model_config = config["model"]["Regression_LSTM"]
        model = Regression_LSTM(model_config=model_config, device=device).to(device).double()

    elif args.model == "Ranking_MLP":
        train_dataset = HandCraftedDataset(config=config, pooling=True, mode="train", pairwise_ranking=True)
        valid_dataset = HandCraftedDataset(config=config, pooling=True, mode="valid", pairwise_ranking=True)

        model_config = config["model"]["Ranking_MLP"]    
        model = RankNet(model_config=model_config, device=device, backbone="MLP").to(device).double()
    
    elif args.model == "Ranking_LSTM":
        train_dataset = HandCraftedDataset(config=config, pooling=False, mode="train", pairwise_ranking=True)
        valid_dataset = HandCraftedDataset(config=config, pooling=False, mode="valid", pairwise_ranking=True)
                
        model_config = config["model"]["Ranking_LSTM"]
        model = RankNet(model_config=model_config, device=device, backbone="LSTM").to(device).double()

    else:
        raise Exception ("Invalid model")

    # tensorboard session
    now_tuple = time.localtime(time.time())
    date_info = '%02d-%02d-%02d_%02d:%02d'%(now_tuple[0]%100,now_tuple[1],now_tuple[2],now_tuple[3],now_tuple[4])
    tensorboard_dir = os.path.join("tensorboard", "train_memorability", args.model, date_info)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print("tensorboard dir located in {}".format(tensorboard_dir))

    writer.add_histogram("LabeledScore/train", train_dataset.filename_memorability_df["score"].values)
    writer.add_histogram("LabeledScore/valid", valid_dataset.filename_memorability_df["score"].values)

    exp_config = config["experiment"]
    train_loader = DataLoader(dataset=train_dataset, batch_size=exp_config["batch_size"], shuffle=True, num_workers=15)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=exp_config["batch_size"], shuffle=False, num_workers=15)
    
    dataiter = iter(train_loader)
    data = dataiter.next()
    # don't know why need to wrap data in another yet, but it works
    writer.add_graph(model, [data])

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=10, verbose=True)

    optimizer = torch.optim.Adam(model.parameters(), model_config["learning_rate"])

    if args.model not in ["Ranking_MLP", "Ranking_LSTM"]:
        ''' not ranking model'''
        smallest_valid_loss = 10000
        for epoch in range(exp_config["epochs"]):
            # train
            model.train()
            batch_train_loss = [] # record the loss of every batch
            train_prediction_list, valid_prediction_list = [], []
            for data in tqdm(train_loader, desc="train"):
                
                optimizer.zero_grad()           
                # sequential_features, non_sequential_features, labels = data
                # prediction, loss = model(sequential_features, non_sequential_features, labels) 
                prediction, loss = model(data) 
                train_prediction_list.append(prediction)
                batch_train_loss.append(loss.cpu().detach().numpy())
                loss.backward() # compute gradient
                optimizer.step()


            writer.add_histogram("PredictedScore/train", torch.cat(train_prediction_list), epoch+1)    
            epoch_train_loss = np.mean(batch_train_loss)
            writer.add_scalar("Loss/train", epoch_train_loss, epoch)
            # validation
            with torch.no_grad():
                model.eval()
                batch_valid_loss = [] # record the loss of every batch
                
                for data in tqdm(valid_loader, desc="valid"):
                    prediction, loss = model(data)
                    valid_prediction_list.append(prediction)
                    batch_valid_loss.append(loss.cpu().detach().numpy())
                    
                writer.add_histogram("PredictedScore/valid", torch.cat(valid_prediction_list), epoch+1)
                epoch_valid_loss = np.mean(batch_valid_loss)
                writer.add_scalar("Loss/valid", epoch_valid_loss, epoch)
                if epoch_valid_loss < smallest_valid_loss:
                    best_model = model
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch+1)
            
            print('\nepoch: {:2}/{},  train loss {:.3f}  valid loss {:.3f}\n'.format(
                                        epoch+1, exp_config["epochs"], epoch_train_loss, epoch_valid_loss))
            
            writer.flush()
            early_stopping(epoch_valid_loss, model)
            if early_stopping.early_stop:
                print("Early Stoppoing")
                break


    else:
        ''' ranking model'''
        smallest_valid_loss = 10000
        for epoch in range(exp_config["epochs"]):
            # train
            model.train()
            batch_train_loss = [] # record the loss of every batch
            batch_train_results = []
            for data in tqdm(train_loader, desc="train"):
    
                optimizer.zero_grad()           
                _, results, loss = model(data) 
                
                batch_train_loss.append(loss.cpu().detach().numpy())
                batch_train_results.append(results.cpu().detach().numpy().squeeze(1))
                loss.backward() # compute gradient
                optimizer.step()
                
            epoch_train_loss = np.mean(batch_train_loss)
            unique, counts = np.unique(np.concatenate(batch_train_results), return_counts=True)
            occ_dict = dict(zip(unique, counts))
            epoch_train_acc = occ_dict[True]/sum([a.shape[0] for a in batch_train_results])
            
            writer.add_scalar("Loss/train", epoch_train_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_train_acc, epoch)

            # validation
            with torch.no_grad():
                model.eval()
                batch_valid_loss = [] # record the loss of every batch
                batch_valid_results = []
                
                for data in tqdm(valid_loader, desc="valid"):
                    _, results, loss = model(data) 
                    batch_valid_loss.append(loss.cpu().detach().numpy())
                    batch_valid_results.append(results.cpu().detach().numpy().squeeze(1))
                    
                epoch_valid_loss = np.mean(batch_valid_loss)

                unique, counts = np.unique(np.concatenate(batch_valid_results), return_counts=True)
                occ_dict = dict(zip(unique, counts))
                epoch_valid_acc = occ_dict[True]/sum([a.shape[0] for a in batch_valid_results])

                writer.add_scalar("Loss/valid", epoch_valid_loss, epoch)
                writer.add_scalar("Accuracy/vakud", epoch_valid_acc, epoch)
                if epoch_valid_loss < smallest_valid_loss:
                    best_model = model

            print('\nepoch: {:2}/{},  train loss {:.3f}  train acc {:.3f}  valid loss {:.3f} valid acc {:.3f}\n'.format(
                epoch+1, exp_config["epochs"], epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc))
            
            writer.flush()
            early_stopping(epoch_valid_loss, model)
            if early_stopping.early_stop:
                print("Early Stoppoing")
                break

    

    model_weight_dir = "weights/train_memorability"
    model_weight_subdir = os.path.join((model_weight_dir), args.model, date_info)
    if not os.path.exists(model_weight_subdir):
        os.makedirs(model_weight_subdir)
    model_weight_path = os.path.join(model_weight_subdir, "{}.pt".format(args.model))
    torch.save(best_model.state_dict(), model_weight_path)
    print("weights of best model '{}' saved at {}".format(args.model, model_weight_path))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train config')
    parser.add_argument("-c", "--config", help="config file location", default="config/engineered_config.yaml")
    parser.add_argument("-m", "--model", help="Regression_MLP, Regression_LSTM, Ranking_LSTM, Ranking_MLP", default="Ranking_MLP")
    parser.add_argument("-e", "--early_stopping", help="early stoppping or not", default=True, type=bool)

    args = parser.parse_args()
    train(args)

    
 