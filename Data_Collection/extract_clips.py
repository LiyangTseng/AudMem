import os
import sys
import argparse
import pandas as pd
import librosa
import soundfile as sf

clip_length = 5

def extract_clip(file_df_row, clip_dir, args):

    ''' using chorus location to make clip '''

    print('==============================')
    print('Extracting progress: {:02}/{}, {}'.format(idx+1, len(intro_df), row['file']))
    audio_file = file_df_row["file"]
    song_wav_data, sr = librosa.load(audio_file)
    filename = audio_file.split('/')[-1][:11]+".wav"

    if not args.fixed_length:
        # variavle length of clip
        output_file = os.path.join(clip_dir, 'beforeNormalize_intro_'+filename)
        if args.audacity:
            print('Enter intro start location with format [min]:[sec]: ', end='')
            intro_start_location = input()
        else:
            intro_start_location = file_df_row["intro_start"]

        minite = int(intro_start_location.split(':')[0])
        second = float(intro_start_location.split(':')[1])
        intro_start = minite*60 + second
        
        if args.audacity:
            print('Enter intro end location with format [min]:[sec]: ', end='')
            intro_end_location = input()
        else:
            intro_end_location = file_df_row["intro_start"]

        minite = int(intro_end_location.split(':')[0])
        second = float(intro_end_location.split(':')[1])
        intro_end = minite*60 + second

    else:
        # fixed length of clip_length sec
        output_file = os.path.join(clip_dir, 'vanilla_5s_intro_'+filename)
        if args.audacity:
            print('Enter intro start location with format [min]:[sec]: ', end='')
            intro_start_location = input()
        else:
            intro_start_location = file_df_row["intro_start"]
        minite = int(intro_start_location.split(':')[0])
        second = float(intro_start_location.split(':')[1])
        intro_start = minite*60 + second
        intro_end = intro_start+clip_length
    
    print('Extracting clip...')
    # clip_wav_data = song_wav_data[int(intro_start*sr) : int((intro_start+clip_length)*sr)]
    clip_wav_data = song_wav_data[int(intro_start*sr) : int((intro_end)*sr)]
    sf.write(output_file, clip_wav_data, sr)
    print('Clip saved at {}'.format(output_file))

if __name__ == '__main__':
    
    ''' usage: python extract_clips.py -index [starting index] -audacity [use audacity or not] '''

    parser = argparse.ArgumentParser(description='Starting location and audacity or not')
    parser.add_argument('-i', '--index', help='starting index (count from 1)', default=1)
    parser.add_argument('-a', '--audacity', help='use audacity or not', default=False, type=bool)
    parser.add_argument('-f', '--fixed_length', help='fixed_length or not', default=False, type=bool)
    parser.add_argument('-s', '--source', help='source of location file', default='more_intro_locations.csv')
    parser.add_argument('-o', '--output', help='directory of output clip', default='Audios/new_raw_variable_intro')
    args = parser.parse_args()

    intro_df = pd.read_csv(args.source)
    file_exist = [os.path.exists(f) for f in intro_df['file']]
    intro_df = intro_df[file_exist]
    intro_df.to_csv(args.source, index=False)
    start_idx = int(args.index)-1

    print(args.audacity)
    input()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    for idx, row in intro_df[start_idx:].iterrows():
        if args.audacity:
            print("open {} with audacity".format(row['file']))
            os.system('audacity "{}"'.format(row['file']))
            # need to close audacity first !!
        extract_clip(file_df_row=row, clip_dir=args.output, args=args)
        