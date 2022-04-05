import os
import json
import numpy as np
import pandas as pd

FOLD_NUM = 10
features_dict = {"chords": ["chroma", "tonnetz"], "timbre": ["mfcc"], "rhythms": ["tempogram"]}

''' this script should be executed in the Model_Architecture directory '''

def calculate_all_features_stats(label_path, features_dir, stats_file_path):
    """ calculate each feature's mean and std in every fold """
    df = pd.read_csv(label_path)
    track_names = df['track_name'].tolist()

    stats_dict = {}
    with open(stats_file_path, 'w') as fp:
        json.dump(stats_dict, fp)

def get_features_stats(label_df, features_dir, for_test=False, features_dict=features_dict):
    """ get feature's mean and std """
    stats_dict = {}
    track_names = label_df.track
    if for_test:
        augmented_type_list = sorted(os.listdir(features_dir))[-1:]
    else:
        augmented_type_list = sorted(os.listdir(features_dir))[:]

    for feature_type in features_dict:
        if feature_type == "emotions":
            continue
        for feature_name in features_dict[feature_type]:
            stats_dict[feature_name] = {}
            features = []
            for track_name in track_names:
                for augmented_type in augmented_type_list:
                    feature_path = os.path.join(features_dir, augmented_type, feature_type, feature_name, "{}_{}".format(feature_name, track_name.replace("wav", "npy")))
                    features.append(np.float32(np.load(feature_path)))
            features = np.array(features)
            assert len(features.shape) == 3, "only calculate mean and std for features with temporal and sptatial dimension"
            stats_dict[feature_name]["mean"] = np.mean(np.mean(features, axis=2), axis=0)
            
            stats_dict[feature_name]["std"] = np.std(np.std(features, axis=2), axis=0)+ 1e-19
    return stats_dict
    

if __name__ == "__main__":
    label_path = "data/labels/track_memorability_scores_beta.csv"
    features_dir = "data/features"
    stats_file_path = os.path.join(features_dir, "stats.json")
    calculate_all_features_stats(label_path, features_dir, stats_file_path)