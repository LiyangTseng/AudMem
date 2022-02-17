'''
    Generate dataset configurations for Pase/Pase+ input. Note that this script only support Audioset and memo data for now.
    (1) To generate data config for audioset in AudMem/Model_Architecture, run:
        python utils/generate_data_cfg.py --task audioset
    (2) To generate data config for memorability data in AudMem/Model_Architecture, run:
        python utils/generate_data_cfg.py --task memo

'''
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import json
import argparse
import torchaudio
import glob

NON_TEST_RATIO = 0.8

def get_file_dur(fname):
    try:
        x, rate = torchaudio.load(fname)
    except RuntimeError:
        print(f"Error processing {fname}")
        return (0)

    return x.shape[1]


def generate_audioset_data_config(opts):
    data_cfg = {'train':{'data':[]},
                'valid':{'data':[]},
                'test':{'data':[]}}

    files = list(glob.glob(opts.data_root+"/*.wav"))
    
    non_test_num = int(len(files) * NON_TEST_RATIO)
    test_files = files[non_test_num:]
    valid_files = files[:int(non_test_num * opts.val_ratio)]
    train_files = files[int(non_test_num * opts.val_ratio):non_test_num]
    
    train_dur = 0
    for idx, train_file in enumerate(train_files, start=1):
        print('Processing train file {:7d}/{:7d}'.format(idx, 
                            len(train_files)), end='\r')

        file_dur = get_file_dur(train_file)
        if file_dur != 0:
            train_dur += file_dur
            data_cfg['train']['data'].append({'filename':train_file})
    data_cfg['train']['total_wav_dur'] = train_dur
    print()

    valid_dur = 0
    for idx, valid_file in enumerate(valid_files, start=1):
        print('Processing valid file {:7d}/{:7d}'.format(idx, 
                            len(valid_files)), end='\r')
        file_dur = get_file_dur(valid_file)
        if file_dur != 0:
            valid_dur += file_dur
            data_cfg['valid']['data'].append({'filename':valid_file})
    data_cfg['valid']['total_wav_dur'] = valid_dur
    print()

    test_dur = 0
    for idx, test_file in enumerate(test_files, start=1):
        print('Processing test file {:7d}/{:7d}'.format(idx, 
                            len(test_files)), end='\r')
        file_dur = get_file_dur(test_file)
        if file_dur != 0:
            test_dur += file_dur
            data_cfg['test']['data'].append({'filename':test_file})
    data_cfg['test']['total_wav_dur'] = test_dur
    print()

    with open(opts.cfg_file, 'w') as cfg_f:
        cfg_f.write(json.dumps(data_cfg))
    print("data config file saved at {}".format(opts.cfg_file))

def generate_memo_data_config(opts):
    data_cfg = {'train':{'data':[]},
                'valid':{'data':[]},
                'test':{'data':[]}}

    memo_df = pd.read_csv("data/labels/track_memorability_scores_beta.csv")
    train_df = memo_df[:200]
    valid_df = memo_df[200:220]
    test_df = memo_df[220:]
    
    train_files, valid_files, test_files = [], [], []
    
    for augmented_type in os.listdir(opts.data_root):
        for file in train_df.track.values:
            train_files.append(os.path.join(opts.data_root, augmented_type, file))
        for file in valid_df.track.values:
            valid_files.append(os.path.join(opts.data_root, augmented_type, file))
    
    for file in test_df.track.values:
            test_files.append(os.path.join(opts.data_root, "original", file))


    train_dur = 0
    for idx, train_file in enumerate(train_files, start=1):
        print('Processing train file {:7d}/{:7d}'.format(idx, 
                            len(train_files)), end='\r')

        data_cfg['train']['data'].append({'filename':train_file})
        train_dur += get_file_dur(train_file)
    data_cfg['train']['total_wav_dur'] = train_dur
    print()

    valid_dur = 0
    for idx, valid_file in enumerate(valid_files, start=1):
        print('Processing valid file {:7d}/{:7d}'.format(idx, 
                            len(valid_files)), end='\r')

        data_cfg['valid']['data'].append({'filename':valid_file})
        valid_dur += get_file_dur(valid_file)
    data_cfg['valid']['total_wav_dur'] = valid_dur
    print()

    test_dur = 0
    for idx, test_file in enumerate(test_files, start=1):
        print('Processing test file {:7d}/{:7d}'.format(idx, 
                            len(test_files)), end='\r')

        data_cfg['test']['data'].append({'filename':test_file})
        test_dur += get_file_dur(test_file)
    data_cfg['test']['total_wav_dur'] = test_dur
    print()

    with open(opts.cfg_file, 'w') as cfg_f:
        cfg_f.write(json.dumps(data_cfg))
    print("data config file saved at {}".format(opts.cfg_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help="audioset, memo", default="audioset")
    parser.add_argument('--val_ratio', type=float, default=0.25,
                        help='Validation ratio to take out of training '
                             'in utterances ratio (Def: 0.1).')
    opts = parser.parse_args()
    if opts.task == "audioset":
        print("processing audioset...")
        setattr(opts, "data_root", '/media/lab812/53D8AD2D1917B29C/audioset/wav')
        setattr(opts, "cfg_file", 'data/audioset/audioset_data.cfg')
        generate_audioset_data_config(opts)
    elif opts.task == "memo":
        print("processing memo...")
        setattr(opts, "data_root", 'data/raw_audios')
        setattr(opts, "cfg_file", 'data/memo_data.cfg')
        generate_memo_data_config(opts)
    else:
        raise Exception("Task not recognize")

        
    

