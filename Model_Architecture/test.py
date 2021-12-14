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
    print("Using device: ", device)
    return device

# def get_pairwise_ranking(args):
def test(args):
    ''' inference ranknet on test dataset, will output all pairwise results to txt file '''
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

        if not os.path.exists(args.memo_score_dir):
            os.makedirs(args.memo_score_dir)
        memorability_score_path = os.path.join(args.memo_score_dir, "predicted_memorability_scores.csv")
        # inference and save outputs to text file
        with open(memorability_score_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["track", "score"])
            for idx, data in enumerate(tqdm(test_loader)):
                score = model(data)
                # ref: https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
                score = score.cpu().detach().item()
                writer.writerow([test_dataset.idx_to_filename[idx], score])
        print("predicted memorability score save at {}".format(memorability_score_path))

        # calculate correlation
        prediction_df = pd.read_csv(memorability_score_path)
        print(stats.spearmanr(prediction_df["score"].values, test_dataset.filename_memorability_df["score"].values))

    elif args.model in ["Ranking_MLP", "Ranking_LSTM"]:
        ''' inference ranknet on test dataset, will output all pairwise results to txt file '''
        
        test_dataset = HandCraftedDataset(config=config, pooling=args.model=="Ranking_MLP", mode="test", pairwise_ranking=True)
        model = RankNet(model_config=model_config, device=device, backbone=args.model[8:], mode="test").to(device).double()
        
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()

        if not os.path.exists(args.ranking_output_dir):
            os.makedirs(args.ranking_output_dir)
        pairwise_ranking_path = os.path.join(args.ranking_output_dir, "pairwise_ranking.csv")
        # inference and save outputs to text file
        with open(pairwise_ranking_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["higher", "lower"])
            for idx in tqdm(range(len(test_dataset.indices_combinations))):
                index_1, index_2 = test_dataset.indices_combinations[idx]
                
                data = test_dataset[idx]
                index_1_higher = model(data)
            
                if index_1_higher:
                    writer.writerow([test_dataset.idx_to_filename[index_1], test_dataset.idx_to_filename[index_2]])
                else:
                    writer.writerow([test_dataset.idx_to_filename[index_2], test_dataset.idx_to_filename[index_1]])
        print("pairwise ranking save at {}".format(pairwise_ranking_path))
        
    else:
        raise Exception ("Invalid model")



def get_overall_ranking(args):
    ''' get pairwise results from txt and compute overall ranking '''
    ground_truth_df =  pd.read_csv("data/labels/track_memorability_scores_beta.csv")[220:].sort_values("score", ascending=False)
    correct_audiofile_order = list(ground_truth_df.track)
    
    filename_to_idx = {correct_audiofile_order[i]: i for i in range(len(correct_audiofile_order))}
    correct_ranking = np.arange(len(correct_audiofile_order))

    ''' calculate overall ranking from pairwise '''
    pairwise_ranking_path = os.path.join(args.ranking_output_dir, "pairwise_ranking.csv")
    with open(pairwise_ranking_path, 'r') as f:
        _ = f.readline() # skip header
        index_pair = []
        for line in f:
            index_pair.append([filename_to_idx[filename] for filename in line.strip().split(',')])
    index_pair = np.array(index_pair)

    # ref: https://stackoverflow.com/questions/51737245/how-to-sort-a-numpy-array-by-frequency
    unique, counts = np.unique(index_pair[:, 0], return_counts=True)     
    sorted_indexes = np.argsort(counts)[::-1]
    sorted_by_freq = unique[sorted_indexes]
    # ref: https://stackoverflow.com/questions/15939748/check-if-each-element-in-a-numpy-array-is-in-another-array
    mask = np.in1d(np.arange(15), sorted_by_freq)
    mininum_idx = np.where(~mask)[0]
    predicted_ranking = np.concatenate((sorted_by_freq, mininum_idx))
    print(predicted_ranking)
    print(stats.spearmanr(correct_ranking, predicted_ranking))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='test config')
    parser.add_argument("-c", "--config", help="config file location", default="config/engineered_config.yaml")
    parser.add_argument("-m", "--model", help="Regression_MLP, Regression_LSTM, Ranking_LSTM, Ranking_MLP", default="Regression_MLP")
    parser.add_argument("-r", "--ranking_output_dir", help="output directory of pairwise ranking", default="output/ranking")
    parser.add_argument("-s", "--memo_score_dir", help="output directory of memorability score", default="output/score")
    parser.add_argument("-o", "--overall", help="get overall ranking or not", default=False, type=bool)

    args = parser.parse_args()

    test(args)
    if args.overall:
        get_overall_ranking(args)


