import pandas as pd
import numpy as np
from scipy import stats

def calculate_views_memorability_relation(memorability_df_path, views_df_path):
    score_df = pd.read_csv(memorability_df_path).sort_values("score")
    filename_to_idx = {list(score_df.track)[i]: i for i in range(len(list(score_df.track)))}
    score_order = np.arange(len(filename_to_idx))

    views_df = pd.read_csv(views_df_path).sort_values("views")
    views_order_list = []
    for track in list(views_df.track):
        views_order_list.append(filename_to_idx[track])
    views_order = np.array(views_order_list)
    # views_order = list(views_df.track)
    return stats.spearmanr(score_order, views_order)

if __name__ == "__main__":
    memorability_df_path = "data/labels/track_memorability_scores_beta.csv"
    views_df_path = "data/track_popularity.csv"
    print(calculate_views_memorability_relation(memorability_df_path, views_df_path))