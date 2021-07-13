import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess(df):
    ''' convert string[123,456] to int 123456'''
    if not pd.api.types.is_numeric_dtype(df['viewCount']):
        df['viewCount'] = df['viewCount'].apply(lambda view: int(view.replace(',', '')))
    return df

def plot_histogram(yt_info_df):

    n_bins = 10

    viewCounts = yt_info_df['viewCount'].to_numpy()
    # viewCounts = np.array([viewCount.replace(',', '') for viewCount in viewCounts]).astype('int')
    log_viewCounts = np.log10(viewCounts)
    log_viewCounts[log_viewCounts == -np.inf] = 0

    plt.figure()
    plt.title('Video Views Distribution')
    plt.hist(log_viewCounts, bins=n_bins, range=[0, 10])
    plt.ylabel('number of videos')
    plt.xlabel('log10 of views')
    plt.savefig('viewCounts.png')

    valid_viewCounts = yt_info_df[yt_info_df['valid']==1]['viewCount'].to_numpy()
    # valid_viewCounts = np.array([valid_viewCount.replace(',', '') for valid_viewCount in valid_viewCounts]).astype('int')
    valid_log_viewCounts = np.log10(valid_viewCounts)
    valid_log_viewCounts[valid_log_viewCounts == -np.inf] = 0

    plt.figure()
    plt.title('Valid Video Views Distribution ')
    plt.hist(valid_log_viewCounts, bins=n_bins, range=[0, 10])
    plt.ylabel('number of videos')
    plt.xlabel('log10 of views')
    plt.savefig('valid_viewCounts.png')

if __name__ == '__main__':
    filename = 'yt_clips.csv'
    yt_info_df = pd.read_csv(filename)
    yt_info_df = preprocess(yt_info_df)
    yt_info_df.to_csv(filename)
    plot_histogram(yt_info_df)
