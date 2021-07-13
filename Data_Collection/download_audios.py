import sys
import os
import pandas as pd

AUDIO_DIR = 'Audios/'

def id_to_audios(vid_id):
    os.system("youtube-dl -f 140 -o '{output_dir}/%(id)s.%(ext)s' https://www.youtube.com/watch?v={id}".format(output_dir=AUDIO_DIR, id=vid_id))

if __name__ == '__main__':
    yt_info_df = pd.read_csv('yt_clips.csv')
    video_num = len(yt_info_df)
    for i in range(video_num):
        print('========== {:02}/{} =========='.format(i+1, video_num))
        yt_vid_id = yt_info_df.iloc[i]['id']
        id_to_audios(yt_vid_id)