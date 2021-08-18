import os
import pandas as pd

clip_dir = '../Audios/clips'
raw_dir = '../Audios/raw'
stats_file = 'stats.csv'

def prune_raw_not_in_clips():
    ''' prune extra raw audios '''
    for raw_file in os.listdir(raw_dir):
        if not os.path.exists(os.path.join(clip_dir, 'clip_'+raw_file)) :
            os.remove(os.path.join(raw_dir, raw_file))
            print('remove file {}'.format(raw_file))

def prune_too_popular(threshold=0):
    ''' prune the videos that have been heard '''
    stats_df = pd.read_csv(stats_file)
    for _, row in stats_df.iterrows():
        popular_score = row['pop_score']
        if popular_score > threshold:
            yt_id = row['url'][-11:]
            os.system('rm {}/{}*'.format(raw_dir, yt_id))
            os.system('rm {}/clip_{}*'.format(clip_dir, yt_id))
            # raw_file_name = row['url'][-11:]+'_'+row['title']+'.wav'
            # raw_file_path = os.path.join(raw_dir, raw_file_name)
            # clip_file_name = 'clip_'+raw_file_name
            # clip_file_path = os.path.join(clip_dir, clip_file_name)
            # os.remove(raw_file_path)
            # os.remove(clip_file_path)
            print('remove {}'.format(row['title']))
        else:
            continue
            
if __name__ == '__main__':
    prune_too_popular(0)