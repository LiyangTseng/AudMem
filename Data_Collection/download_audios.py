import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from plot_statistics import get_histogram

SAMPLED_YT_INFO_FILE = 'temp.csv'
NEEDED_YT_INFO_FILE = 'needed_yt_info.csv'

def sample_data(yt_info_df, exist_distri = np.array([0, 0, 0, 0, 0, 0, 0]), existing_file_list=[]):
    ''' sample video data evenly from yt_info respect to views'''
    
    valid_yt_info_df = yt_info_df[yt_info_df['valid']==1]
    valid_yt_info_df.sort_values(by=['viewCount'], inplace=True)

    sampled_info_df = pd.DataFrame(columns=['id', 'title', 'viewCount'])

    start_exp, end_exp = 2, 9
    exp_target = start_exp
    distri_cnt = 0
    distri_arr = np.array([13, 13, 14, 14, 14, 13, 13]) - exist_distri
    print("needed distribution: ", distri_arr)
    for _, row in valid_yt_info_df.iterrows():
        exp = int(np.log10(row['viewCount']))
        '''
            filter out title with chinise words
            ref: https://stackoverflow.com/questions/34587346/python-check-if-a-string-contains-chinese-character
        '''
        if exp == exp_target and len(re.findall(r'[\u4e00-\u9fff]+', row['title']))==0 and 'https://www.youtube.com/watch?v='+row['id'][-11:] not in existing_file_list:
            sampled_info_df = sampled_info_df.append(row, ignore_index=True)
            distri_cnt += 1
            if distri_cnt == distri_arr[exp-start_exp]:
                exp_target += 1
                distri_cnt = 0
                if exp_target == end_exp:
                    break
                elif distri_arr[exp_target-start_exp] == 0:
                    exp_target += 1 
        else:
            continue
    return sampled_info_df

if __name__ == '__main__':

    ''' python download_audios.py -sample [True/False] '''
    
    parser = argparse.ArgumentParser(description='resample audios or not')
    parser.add_argument('-t', '--type', help='options to download audios', default='complete')
    parser.add_argument('-o', '--output', help='new audio save location', default='raw3')
    parser.add_argument('-e', '--existing', help='existing audio save location', default='raw2')
    args = parser.parse_args()

    if args.type == 'resample':
        ''' resample from yt_info.csv and then download audios  '''
        yt_info_df = pd.read_csv('yt_info.csv')
        sampled_yt_info = sample_data(yt_info_df)
        sampled_yt_info.to_csv(SAMPLED_YT_INFO_FILE, index=False)
        print('Resampled youtube info saved at {}'.format(SAMPLED_YT_INFO_FILE))
        get_histogram(sampled_yt_info, type_name='Sampled')

    elif args.type == 'from_file':
        ''' download audios from given files'''
        sampled_yt_info = pd.read_csv(SAMPLED_YT_INFO_FILE)
        get_histogram(sampled_yt_info, type_name='Sampled')

    elif args.type == 'complete':
        ''' fill up the remaining slots ''' 
        existing_yt_info = pd.read_csv(SAMPLED_YT_INFO_FILE)
        # prune sampled yt_info csv file   
        EXI_AUDIO_DIR = os.path.join('Audios', args.existing)
        existing_file_list = os.listdir(EXI_AUDIO_DIR)
        existing_file_list = os.listdir('Audios/raw') + os.listdir('Audios/raw2')

        existing_file_list = ['https://www.youtube.com/watch?v='+yt_id[:11] for yt_id in existing_file_list]
        existing_yt_info = existing_yt_info[existing_yt_info['id'].isin(existing_file_list)]
        existing_yt_info.to_csv(SAMPLED_YT_INFO_FILE, index=False)

        existing_views = existing_yt_info['viewCount'].to_numpy()
        _, counts = np.unique(get_histogram(existing_views).astype(int), return_counts=True)
        
        yt_info_df = pd.read_csv('yt_info.csv')
        sampled_yt_info = sample_data(yt_info_df, exist_distri=counts, existing_file_list=existing_file_list)
        sampled_yt_info.to_csv(NEEDED_YT_INFO_FILE, index=False)
        print('needed youtube info saved at {}'.format(NEEDED_YT_INFO_FILE))
        needed_views = sampled_yt_info['viewCount'].to_numpy()
        get_histogram(needed_views, type_name='Needed')
    
    SAVE_AUDIO_DIR = os.path.join('Audios', args.output)
    if not os.path.exists(SAVE_AUDIO_DIR):
        os.makedirs(SAVE_AUDIO_DIR)
    
    
    audio_list = os.listdir(SAVE_AUDIO_DIR)
    audio_list = [f[:11] for f in audio_list]

    video_num = len(sampled_yt_info)
    print('Downloading {} audios'.format(video_num))
    for i in range(video_num):
        yt_id = sampled_yt_info.iloc[i]['id'][-11:]
        print('========== {:02}/{} =========='.format(i+1, video_num))
        audio_filename = '{id}_{title}.{ext}'.format(
            id=yt_id, title=sampled_yt_info.iloc[i]['title'], ext='wav')
        if yt_id in audio_list:
            print('already saved')
        else:
            os.system("youtube-dl --extract-audio  --audio-format wav -o '{output_dir}/%(id)s_%(title)s.%(ext)s' {url}".format(
                        output_dir=SAVE_AUDIO_DIR, url=sampled_yt_info.iloc[i]['id']))
