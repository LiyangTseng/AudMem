from pychorus.pychorus.helpers import find_and_output_chorus
import os
import csv
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import argparse


def detect_chorus(arguments, chorus_length=10):
    AUDIO_DIR = arguments.source
    title_chorus_dict = {}
    for audio in tqdm(os.listdir(AUDIO_DIR)):
        audio_path = os.path.join(AUDIO_DIR, audio)
        if audio_path[-3:] in ['mp3', 'wav']:
            chorus_path = '.'        
            chorus_start_sec = find_and_output_chorus(audio_path, chorus_path, chorus_length)
            if chorus_start_sec == None:
                print('could not detect chorus in {}'.format(audio_path))
                title_chorus_dict[audio_path] = ''
                continue
            title_chorus_dict[audio_path] = '{:02}:{:02}'.format(int(chorus_start_sec // 60), int(chorus_start_sec % 60))
        
    with open(arguments.output, 'w') as f:  
        writer = csv.writer(f)
        writer.writerow(['file', 'chorus_location'])
        for key, value in title_chorus_dict.items():
            writer.writerow([key, value])  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='resample audios or not')
    parser.add_argument('-s', '--source', help='directory that stores audios', default='Audios/raw3')
    parser.add_argument('-o', '--output', help='output file of chorus location', default='chorus_location_3.csv')
    args = parser.parse_args()

    detect_chorus(args)