import os
import csv
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import DataLoader
from dataset import HandCraftedDataset
from model.memorability_model import Regression_MLP, Regression_LSTM, RankNet

def check_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    print("[INFO]", "Using device: ", device)
    return device

def test(args):

    device = check_device()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_config = config["model"][args.model]
    model_weight_dir = "weights/train_memorability"
    model_weight_path = os.path.join((model_weight_dir), args.model, "{}.pt".format(args.model))
    
    if args.model in ["Regression_MLP", "Regression_LSTM"]:
        test_dataset = HandCraftedDataset(config=config, pooling=args.model=="Regression_MLP", mode="test")
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=15)
        if args.model == "Regression_MLP":
            model = Regression_MLP(model_config=model_config, device=device, mode="test").to(device).double()
        else:
            model = Regression_LSTM(model_config=model_config, device=device, mode="test").to(device).double()

        model.load_state_dict(torch.load(model_weight_path))
        model.eval()

        # inference and save outputs to text file
        if not os.path.exists(args.memo_score_dir):
            os.makedirs(args.memo_score_dir)
        memorability_score_path = os.path.join(args.memo_score_dir, "predicted_memorability_scores.csv")
        with open(memorability_score_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["track", "score"])
            for idx, data in enumerate(tqdm(test_loader)):
                score = model(data)
                # ref: https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
                score = score.cpu().detach().item()
                writer.writerow([test_dataset.idx_to_filename[idx], score])
        print("[INFO]", "predicted memorability score save at {}".format(memorability_score_path))

        # calculate correlation
        prediction_df = pd.read_csv(memorability_score_path)
        print(stats.spearmanr(prediction_df["score"].values, test_dataset.filename_memorability_df["score"].values))
        
    else:
        raise Exception ("Invalid model")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='test config')
    parser.add_argument("-c", "--config", help="config file location", default="config/engineered_config.yaml")
    parser.add_argument("-m", "--model", help="Regression_MLP, Regression_LSTM", default="Regression_LSTM")
    parser.add_argument("-r", "--ranking_output_dir", help="output directory of pairwise ranking", default="output/ranking")
    parser.add_argument("-s", "--memo_score_dir", help="output directory of memorability score", default="output/score")

    args = parser.parse_args()

    test(args)


