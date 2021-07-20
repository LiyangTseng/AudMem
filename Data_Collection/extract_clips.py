import os
import sys
import pandas as pd
import librosa
import soundfile as sf

CLIP_DIR = 'Audios/clips'
clip_length = 10

def extract_clips(audio_file, chorus_location):
    ''' using chorus location to make clip '''
    print('Extracting clip...')
    song_wav_data, sr = librosa.load(audio_file)
    
    minite = int(chorus_location.split(':')[0])
    second = float(chorus_location.split(':')[1])
    chorus_start = minite*60 + second
    
    filename = audio_file.split('/')[-1]
    if not os.path.exists(CLIP_DIR):
        os.makedirs(CLIP_DIR)
    output_file = os.path.join(CLIP_DIR, 'clip_'+filename)
    chorus_wave_data = song_wav_data[int(chorus_start*sr) : int((chorus_start+clip_length)*sr)]
    sf.write(output_file, chorus_wave_data, sr)
    print('Clip saved at {}'.format(output_file))

if __name__ == '__main__':
    chorus_df = pd.read_csv('chorus_location.csv')
    start_idx = int(sys.argv[1])-1
    
    for idx, row in chorus_df[start_idx:].iterrows():
        os.system('audacity "{}"'.format(row['file']))
        # need to close audacity first !!
        print('==============================')
        print('Editing progress: {:02}/{}, {}'.format(idx+1, len(chorus_df), row['file']))
        print('Enter chorus location with format [min]:[sec]: ', end='')
        chorus_loc = input()
        extract_clips(row['file'], chorus_loc)
