import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

AUDIO_DIR = 'Audios/raw'

def sample_data(yt_info_df):
    ''' sample video data evenly from yt_info respect to views'''
    
    valid_yt_info_df = yt_info_df[yt_info_df['valid']==1]
    valid_yt_info_df.sort_values(by=['viewCount'], inplace=True)

    sampled_info_df = pd.DataFrame(columns=['id', 'title', 'viewCount'])

    start_exp, end_exp = 2, 9
    exp_target = start_exp
    distri_cnt = 0
    distri_list = [13, 13, 14, 14, 14, 13, 13]
    for _, row in valid_yt_info_df.iterrows():
        exp = int(np.log10(row['viewCount']))
        '''
            filter out title with chinise words
            ref: https://stackoverflow.com/questions/34587346/python-check-if-a-string-contains-chinese-character
        '''
        if exp == exp_target and len(re.findall(r'[\u4e00-\u9fff]+', row['title']))==0:
            sampled_info_df = sampled_info_df.append(row, ignore_index=True)
            distri_cnt += 1
            if distri_cnt == distri_list[exp-start_exp]:
                exp_target += 1
                distri_cnt = 0
                if exp_target == end_exp:
                    break
        else:
            continue
    return sampled_info_df

if __name__ == '__main__':
    yt_info_df = pd.read_csv('yt_info.csv')
    sampled_yt_info = sample_data(yt_info_df)
    sampled_yt_info.to_csv('sampled_yt_info.csv', index=False)
    
    n_bins = 7
    viewCounts = sampled_yt_info['viewCount'].to_numpy()
    log_viewCounts = np.log10(viewCounts.astype('float'))

    plt.figure()
    plt.title('Sampled Video Views Distribution')
    plt.hist(log_viewCounts, bins=n_bins, range=[2, 9])
    plt.ylabel('number of videos')
    plt.xlabel('log10 of views')
    plt.savefig('sampled_viewCounts.png')

    
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
        
    video_num = len(sampled_yt_info)
    for i in range(video_num):
        print('========== {:02}/{} =========='.format(i+1, video_num))
        audio_filename = '{id}_{title}.{ext}'.format(
            id=sampled_yt_info.iloc[i]['id'][-11:], title=sampled_yt_info.iloc[i]['title'], ext='wav')
        if not os.path.exists(os.path.join(AUDIO_DIR, audio_filename)):
            os.system("youtube-dl --extract-audio  --audio-format wav -o '{output_dir}/%(id)s_%(title)s.%(ext)s' {url}".format(
                        output_dir=AUDIO_DIR, filename = audio_filename, url=sampled_yt_info.iloc[i]['id']))
