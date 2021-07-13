from pychorus.pychorus.helpers import find_and_output_chorus
import os
import csv
import warnings
warnings.filterwarnings('ignore')

clip_length = 10
audio_dir = 'Audios/raw'

title_chorus_dict = {}
for playlist in os.listdir(audio_dir):
    for song in os.listdir(os.path.join(audio_dir, playlist)):
        audio_path = os.path.join(audio_dir, playlist, song)
        if audio_path[-3:] in ['mp3', 'wav']:
            chorus_path = '.'        
            chorus_start_sec = find_and_output_chorus(audio_path, chorus_path, clip_length)
            if chorus_start_sec == None:
                continue
            title_chorus_dict[audio_path] = '{:02}:{:02}'.format(int(chorus_start_sec // 60), int(chorus_start_sec % 60))
            print('{song} chorus detected!'.format(song=audio_path))  
    
with open('chorus_location.csv', 'w') as f:  
    writer = csv.writer(f)
    for key, value in title_chorus_dict.items():
        writer.writerow([key, value])  
    