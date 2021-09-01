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

def get_histogram(views, type_name='Orignal', n_bins=10, plot=True):

    log_views = np.log10(views.astype('float'))
    log_views[log_views == -np.inf] = 0
    if plot:
        plt.figure()
        plt.title('{} Video Views Distribution'.format(type_name))
        plt.hist(log_views, bins=n_bins, range=[0, 10])
        plt.ylabel('number of videos')
        plt.xlabel('log10 of views')
        plt.savefig('{}_viewCounts.png'.format(type_name.lower()))
        print('Distribution saved at {}_viewCounts.png'.format(type_name.lower()))
    
    ''' in case other module need the distribution information '''
    return log_views

if __name__ == '__main__':
    filename = 'yt_info.csv'
    yt_info_df = pd.read_csv(filename)
    yt_info_df = preprocess(yt_info_df)
    yt_info_df.to_csv(filename, index=False)

    orginal_viewCount = yt_info_df['viewCount'].to_numpy()
    get_histogram(orginal_viewCount)
    valid_viewCounts = yt_info_df[yt_info_df['valid']==1]['viewCount'].to_numpy()
    get_histogram(valid_viewCounts, type_name='Valid')

