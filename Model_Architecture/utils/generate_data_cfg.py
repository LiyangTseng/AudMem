import json
#import librosa
import argparse
import random
from random import shuffle
import numpy as np
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


def main(opts):
    random.seed(opts.seed)
    data_cfg = {'train':{'data':[]},
                'valid':{'data':[]},
                'test':{'data':[]}}

    if opts.train_scp == None and opts.test_scp == None:
        files = list(glob.glob(opts.data_root+"/*.wav"))
        
        non_test_num = int(len(files) * NON_TEST_RATIO)
        test_files = files[non_test_num:]
        valid_files = files[:int(non_test_num * opts.val_ratio)]
        train_files = files[int(non_test_num * opts.val_ratio):non_test_num]
        
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

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                        default='data/audioset/wav')
    parser.add_argument('--train_scp', type=str, default=None)
    parser.add_argument('--valid_scp', type=str, default=None)
    parser.add_argument('--test_scp', type=str, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.25,
                        help='Validation ratio to take out of training '
                             'in utterances ratio (Def: 0.1).')
    parser.add_argument('--cfg_file', type=str, default='data/audioset/audioset_data.cfg')
    parser.add_argument('--seed', type=int, default=3)
    
    opts = parser.parse_args()
    main(opts)

