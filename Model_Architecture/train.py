import warnings
warnings.filterwarnings("ignore")
import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import HandCraftedDataset
from model.engineered_model import MLP

def train(args):
    
    with open(args.config, 'r') as f:
        hyper_parameters = yaml.load(f)
    
    train_engineered_set = HandCraftedDataset(features_dir=args.feature_dir, labels_dir=args.label_dir, mode="train")
    valid_engineered_set = HandCraftedDataset(features_dir=args.feature_dir, labels_dir=args.label_dir, mode="valid")
    test_engineered_set = HandCraftedDataset(features_dir=args.feature_dir, labels_dir=args.label_dir, mode="test")
    
    train_loader = DataLoader(dataset=train_engineered_set, batch_size=hyper_parameters["experiment"]["batch_size"], shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_engineered_set, batch_size=hyper_parameters["experiment"]["batch_size"], shuffle=False, num_workers=2)

    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu") 
    device = "cpu"
    print("Using device: ", device)

    model = MLP().to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), hyper_parameters["experiment"]["learning_rate"])
    loss_function = nn.BCELoss()


    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []

    for epoch in range(hyper_parameters["experiment"]["epochs"]):
        # train
        model.train() # activate train mode
        for _, data in enumerate(train_loader):
            features, labels = data
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad() # clear previous gradient
            outputs = model(features) # feed data into model
            predictions = torch.max(outputs, 1)[1] # return the indices for the max through axis 1 
            acc = (predictions == labels).sum() / features.shape[0]
            loss = loss_function(outputs, labels.reshape(-1,1))
            loss.backward() # compute gradient
            optimizer.step()
            # training_accuracy.append(acc)
            # training_loss.append(loss)

        training_accuracy.append(acc)
        training_loss.append(loss)
        
        # validation
        with torch.no_grad():
            model.eval()
            for _, data in enumerate(valid_loader):
                features, labels = data
                features, labels = features.to(device), labels.to(device)
                outputs = model(features) # feed data into model
                predictions = torch.max(outputs, 1)[1] # return the indices for the max through axis 1 
                acc = (predictions == labels).sum() / features.shape[0]
                loss = loss_function(outputs, labels.reshape(-1,1))
                # validation_accuracy.append(acc)
                # validation_loss.append(loss)
                
                validation_accuracy.append(acc)
                validation_loss.append(loss)
        print('epoch: {:2},  training loss {:.2f}  training acc: {:.2f}  validation loss {:.2f}  validation acc: {:.2f}'.format(
            epoch+1, training_loss[epoch], training_accuracy[epoch], validation_loss[epoch], validation_accuracy[epoch]))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train config')
    parser.add_argument("-p", "--preprocess", help="preprocessing or not", default=False, type=bool)
    parser.add_argument("-c", "--config", help="config file location", default="config/engineered.yaml")
    parser.add_argument("-f", "--feature_dir", help="directory of features", default="../Feature_Extraction/features")
    parser.add_argument("-l", "--label_dir", help="directory of labels", default="../Feature_Extraction/labels")

    args = parser.parse_args()
    train(args)

    
 