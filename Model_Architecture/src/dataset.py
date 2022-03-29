import os
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import json
from itertools import combinations
from src.transforms import *
from src.util import _prepare_weights
import sys

class HandCraftedDataset(Dataset):
    ''' Hand crafted audio features to predict memorability (classification) '''
    def __init__(self, labels_df, config, pooling=False, split="train"):
        super().__init__()
        assert split in ["train", "valid", "test"], "invalid split"
        self.labels_df = labels_df
        self.split = split

        self.features_dir = config["path"]["features_dir"]
        
        if self.split != "test":
            # self.augmented_type_list = sorted(os.listdir(self.features_dir))[-1:]
            self.augmented_type_list = sorted(os.listdir(self.features_dir))[:]
        else:
            self.augmented_type_list = sorted(os.listdir(self.features_dir))[-1:]
        self.pooling = pooling

        self.track_names = list(self.labels_df.track)
        self.scores = list(self.labels_df.score)
        self.filename_to_score = dict(zip(self.track_names, self.scores))


        self.features_dict = config["features"]

        self.features_options = [[] for _ in range(len(self.labels_df))]

        for track_name in tqdm(self.track_names):
            for augment_type in self.augmented_type_list:
                
                sequeutial_features_list = []
                non_sequential_features_list = []
                
                for feature_type in self.features_dict:
                    for subfeatures in self.features_dict[feature_type]:
                        feature_file_path = os.path.join(self.features_dir, augment_type, feature_type, subfeatures,
                             "{}_{}".format(subfeatures, track_name.replace("wav", "npy")))   
                        f = np.load(feature_file_path)
                        f = np.float32(f)
                        if feature_type == "emotions":
                            # features not sequential
                            non_sequential_features_list.append(f)
                        else:
                            # features are sequential 
                            if self.pooling:
                                # temporal mean pooling to compress 2d features to 1d
                                sequeutial_features_list.append(f.mean(axis=1))
                                sequential_features = np.concatenate(sequeutial_features_list, axis=0)
                            else:
                                sequeutial_features_list.append(f)
                                sequential_features = np.concatenate(sequeutial_features_list, axis=0)
                                sequential_features = np.transpose(sequential_features)   

                sequential_features = torch.from_numpy(sequential_features)
                non_sequential_features = torch.from_numpy(np.stack(non_sequential_features_list))
                
                
                self.features_options[self.track_names.index(track_name)].append(
                        [sequential_features, non_sequential_features]
                    )        

    def __len__(self):
        return len(self.labels_df)*len(self.augmented_type_list)   

    def __getitem__(self, index):
        features_options = self.features_options[index]
        features = features_options[np.random.randint(0, len(features_options))]
        score = self.scores[index]

        return features, score


class PairHandCraftedDataset(HandCraftedDataset):
    ''' Hand crafted audio features to predict memorability (classification) '''
    def __init__(self, labels_df, config, pooling=False, split="train"):
        super().__init__(labels_df=labels_df, config=config, pooling=pooling, split=split)
        assert split in ["train", "valid"], "Split must be either train or valid"
        # get index combinations of wavs
        self.index_combinations = list(combinations([i for i in range(len(self.labels_df))], 2))
        self.feature_combinations, self.score_combinations = [], []
        for index_pair in self.index_combinations:
            index_1, index_2 = index_pair
            self.feature_combinations.append([self.features_options[index_1],
                                            self.features_options[index_2]])
            self.score_combinations.append([self.scores[index_1],
                                            self.scores[index_2]])


    def __len__(self):
        return len(self.index_combinations)

    def __getitem__(self, index):
        features_options_1, features_options_2 = self.feature_combinations[index]
        features_1 = features_options_1[np.random.randint(0, len(features_options_1))]
        features_2 = features_options_2[np.random.randint(0, len(features_options_2))]
        score_1, score_2 = self.score_combinations[index]

        return features_1, features_2, score_1, score_2


class DictCollater(object):

    def __init__(self, batching_keys=['cchunk',
                                      'chunk',
                                      'chunk_ctxt',
                                      'chunk_rand',
                                      'overlap',
                                      'lps',
                                      'lpc',
                                      'gtn',
                                      'fbank',
                                      'mfcc',
                                      'mfcc_librosa',
                                      'prosody',
                                      'kaldimfcc',
                                      'kaldiplp'],
                 meta_keys=[],
                 labs=False):
        self.batching_keys = batching_keys
        self.labs = labs
        self.meta_keys = meta_keys

    def __call__(self, batch):
        batches = {}
        lab_b = False
        labs = None
        lab_batches = []
        meta = {}
        for sample in batch:
            if len(sample) > 1 and self.labs:
                labs = sample[1:]
                sample = sample[0]
                if len(lab_batches) == 0:
                    for lab in labs:
                        lab_batches.append([])
            for k, v in sample.items():
                if k in self.meta_keys:
                    if k not in meta:
                        meta[k] = []
                    meta[k].append(v)
                if k not in self.batching_keys:
                    continue
                if k not in batches:
                    batches[k] = []
                if v.dim() == 1:
                    v = v.view(1, 1, -1)
                elif v.dim() == 2:
                    v = v.unsqueeze(0)
                else:
                    raise ValueError('Error in collating dimensions for size '
                                     '{}'.format(v.size()))
                batches[k].append(v)
            if labs is not None:
                for lab_i, lab in enumerate(labs):
                    lab_batches[lab_i].append(lab)
        for k in batches.keys():
            batches[k] = torch.cat(batches[k], dim=0)
        rets = [batches]
        if labs is not None:
            for li in range(len(lab_batches)):
                lab_batches_T = lab_batches[li]
                lab_batches_T = torch.tensor(lab_batches_T)
                rets.append(lab_batches_T)
        if len(meta) > 0:
            rets.append(meta)
        if len(rets) == 1:
            return rets[0]
        else:
            return rets

class WavDataset(Dataset):
    ''' source: https://github.com/santi-pdp/pase '''

    def __init__(self, data_root, data_cfg_file, split,
                 transform=None, sr=None,
                 preload_wav=False,
                 transforms_cache=None,
                 distortion_transforms=None,
                 cache_on_load=False,
                 distortion_probability=0.4,
                 verbose=True,
                 same_sr = False,
                 *args, **kwargs):
        # sr: sampling rate, (Def: None, the one in the wav header)
        self.sr = sr
        self.data_root = data_root
        self.cache_on_load = cache_on_load
        self.data_cfg_file = data_cfg_file
        if not isinstance(data_cfg_file, str):
            raise ValueError('Please specify a path to a cfg '
                             'file for loading data.')

        self.split = split
        self.transform = transform
        self.transforms_cache = transforms_cache
        self.distortion_transforms = distortion_transforms
        self.preload_wav = preload_wav
        self.distortion_probability = distortion_probability
        self.same_sr = same_sr
        with open(data_cfg_file, 'r') as data_cfg_f:
            self.data_cfg = json.load(data_cfg_f)
            
            if split != "all":
                wavs = self.data_cfg[split]['data']
                self.total_wav_dur = int(self.data_cfg[split]['total_wav_dur'])
                self.wavs = wavs
            else:
                self.total_wav_dur = 0
                self.wavs = []
                for _split in self.data_cfg:
                    wavs = self.data_cfg[_split]['data']
                    self.total_wav_dur += int(self.data_cfg[_split]['total_wav_dur'])
                    self.wavs += wavs
        self.wav_cache = {}
        if preload_wav:
            print('Pre-loading wavs to memory')
            for wavstruct in tqdm(self.wavs, total=len(self.wavs)):
                uttname = wavstruct['filename']
                wname = os.path.join(self.data_root, uttname)
                self.retrieve_cache(wname, self.wav_cache)

    def __len__(self):
        return len(self.wavs)

    def retrieve_cache(self, fname, cache):
        if (self.cache_on_load or self.preload_wav) and fname in cache:
            return cache[fname]
        else:
            wav, rate = torchaudio.load(fname)
            if self.same_sr:
                assert self.sr!=None, "sampling rate must be specified"
                wav = torchaudio.transforms.Resample(rate, self.sr)(wav)
            wav = wav.numpy().squeeze()
            #fix in case wav is stereo, in which case
            #pick first channel only
            if wav.ndim > 1:
                wav = wav[0, :]
            wav = wav.astype(np.float32)
            if self.cache_on_load:
                cache[fname] = wav
            return wav

    def __getitem__(self, index):
        # uttname = self.wavs[index]['filename']
        # wname = os.path.join(self.data_root, uttname)
        wname = self.wavs[index]['filename']
        wav = self.retrieve_cache(wname, self.wav_cache)
        if self.transform is not None:
            wav = self.transform(wav)
        wav['cchunk'] = wav['chunk'].squeeze(0)

        return wav

class MemoWavDataset(WavDataset):
    '''
    Note that data augmentation is done by randomly selecting shifting version of the wav when calling __getitem__,
    hence the length of this dataset is the length of the label dataframe.
    '''
    def __init__(self, labels_df, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_df = labels_df
        self.track_names = list(self.labels_df.track)
            
        self.filename_to_score = dict(zip(self.labels_df.track, self.labels_df.score))
        
        # self.filename_options = dict.fromkeys(self.filename_to_score.keys(), [])
        self.filename_options = [[] for _ in range(len(self.filename_to_score))]

        if self.split == "test":
            for wname in self.track_names:
                self.filename_options[self.track_names.index(wname)].append(
                        "data/raw_audios/original/{}".format(wname))
        else:
            self.augment_types = ["1_semitones", "2_semitones", "3_semitones", "4_semitones",
                        "5_semitones", "-1_semitones","-2_semitones", 
                        "-3_semitones", "-4_semitones", "-5_semitones", "original"]
            for track_name in self.track_names:
                for augment_type in self.augment_types:
                    self.filename_options[self.track_names.index(track_name)].append(
                        "data/raw_audios/{}/{}".format(augment_type, track_name))

        self.scores = []
        for idx in range(len(self.labels_df)):
            self.scores.append(self.filename_to_score[list(self.labels_df.track)[idx]])
        # for wav in self.wavs:
        #     wname = os.path.join(self.data_root, wav["filename"])
        #     wname = wname.split("/")[-1]
        #     self.scores.append(self.filename_to_score[wname])

    def __len__(self):
        return len(self.filename_to_score)

    def __getitem__(self, index):
        # track_name = self.labels_df.track[index]
        # wav_options = self.filename_options[track_name]
        # score = self.filename_to_score[track_name]
        wav_options = self.filename_options[index]
        score = self.scores[index]
        wname = wav_options[np.random.randint(0, len(wav_options))]
        wav = self.retrieve_cache(wname, self.wav_cache)

        return wav, score

class PairMemoWavDataset(WavDataset):
    def __init__(self, labels_df, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.split!="test", "pair dataset only for train and valid"
        self.labels_df = labels_df
        self.track_names = list(self.labels_df.track)
        self.scores = list(self.labels_df.score)
        self.filename_to_score = dict(zip(self.track_names, self.scores))
        # self.filename_options = dict.fromkeys(self.filename_to_score.keys(), [])
        self.filename_options = [[] for _ in range(len(self.labels_df))]

        augment_types = ["1_semitones", "2_semitones", "3_semitones", "4_semitones",
                        "5_semitones", "-1_semitones","-2_semitones", 
                        "-3_semitones", "-4_semitones", "-5_semitones", "original"]
        for track_name in self.track_names:
            for augment_type in augment_types:
                self.filename_options[self.track_names.index(track_name)].append(
                    "data/raw_audios/{}/{}".format(augment_type, track_name))


        # get index combinations of wavs
        self.index_combinations = list(combinations([i for i in range(len(self.labels_df))], 2))
        self.wav_combinations, self.score_combinations = [], []
        for index_pair in self.index_combinations:
            index_1, index_2 = index_pair
            self.wav_combinations.append([self.filename_options[index_1],
                                            self.filename_options[index_2]])
            self.score_combinations.append([self.scores[index_1],
                                            self.scores[index_2]])


    def __len__(self):
        return len(self.index_combinations)

    def __getitem__(self, index):
        
        wav_options_1, wav_options_2 = self.wav_combinations[index]
        wname_1 = wav_options_1[np.random.randint(0, len(wav_options_1))]
        wname_2 = wav_options_2[np.random.randint(0, len(wav_options_2))]
        wav_1, wav_2 = self.retrieve_cache(wname_1, self.wav_cache), self.retrieve_cache(wname_2, self.wav_cache)
        score_1, score_2 = self.score_combinations[index]

        return wav_1, wav_2, score_1, score_2

def build_dataset_providers(config, minions_cfg):

    dr = len(config["path"]["data_root"])
    dc = len(config["path"]["data_cfg"])

    if dr > 1 or dc > 1:
        assert dr == dc, (
            "Specced at least one repeated option for data_root or data_cfg."
            "This assumes multiple datasets, and their resp configs should be matched."
            "Currently got {} data_root and {} data_cfg options".format(dr, dc)
        )



    if len(config["dataset"]["dataset_list"]) < 1:
        config["dataset"]["dataset_list"].append('WavDataset')

    #TODO: allow for different base transforms for different datasets
    trans, batch_keys = make_transforms(config["dataset"]["chunk_size"], minions_cfg,
                                        config["dataset"]["hop"],
                                        config["dataset"]["random_scale"],
                                        config["path"]["stats"], config["dataset"]["trans_cache"])

    dsets, va_dsets = [], []
    for idx in range(dr):
        # print ('Preparing dset for {}'.format(config["path"]["data_root"][idx]))
        

        # Build Dataset(s) and DataLoader(s), ref: https://stackoverflow.com/questions/2933470/how-do-i-call-setattr-on-the-current-module
        dataset = getattr(sys.modules[__name__], config["dataset"]["dataset_list"][idx])
        dset = dataset(config["path"]["data_root"][idx], config["path"]["data_cfg"][idx], 
                       'train',
                       transform=trans,
                       distortion_transforms=None,
                       preload_wav=config["dataset"]["preload_wav"])

        dsets.append(dset)

        if config["dataset"]["do_eval"]:
            va_dset = dataset(config["path"]["data_root"][idx], config["path"]["data_cfg"][idx],
                              'valid', 
                              transform=trans,
                              distortion_transforms=None,
                              preload_wav=config["dataset"]["preload_wav"])
            va_dsets.append(va_dset)

    ret = None
    if len(dsets) > 1:
        ret = (MetaWavConcatDataset(dsets), )
        if config["dataset"]["do_eval"]:
            ret = ret + (MetaWavConcatDataset(va_dsets), )
    else:
        ret = (dsets[0], )
        if config["dataset"]["do_eval"]:
            ret = ret + (va_dsets[0], )

    if config["dataset"]["do_eval"] is False or len(va_dsets) == 0:
        ret = ret + (None, )

    return ret, batch_keys

def make_transforms(chunk_size, workers_cfg, hop,
                    random_scale=False,
                    stats=None, trans_cache=None):
    trans = [ToTensor()]
    keys = ['totensor']

    trans.append(SingleChunkWav(chunk_size, random_scale=random_scale))

    collater_keys = []
    znorm = False
    for type, minions_cfg in workers_cfg.items():
        for minion in minions_cfg:
            name = minion['name']
            if name in collater_keys:
                raise ValueError('Duplicated key {} in minions'.format(name))
            collater_keys.append(name)
            # look for the transform config if available 
            # in this minion
            tr_cfg=minion.pop('transform', {})
            tr_cfg['hop'] = hop
            if name == 'mi' or name == 'cmi' or name == 'spc' or \
               name == 'overlap' or name == 'gap' or 'regu' in name:
                continue
            elif 'lps' in name:
                znorm = True
                # copy the minion name into the transform name
                tr_cfg['name'] = name
                #trans.append(LPS(opts.nfft, hop=opts.LPS_hop, win=opts.LPS_win, der_order=opts.LPS_der_order))
                trans.append(LPS(**tr_cfg))
            elif 'gtn' in name:
                znorm = True
                tr_cfg['name'] = name
                trans.append(Gammatone(**tr_cfg))
                #trans.append(Gammatone(opts.gtn_fmin, opts.gtn_channels, 
                #                       hop=opts.gammatone_hop, win=opts.gammatone_win,der_order=opts.gammatone_der_order))
            elif 'lpc' in name:
                znorm = True
                tr_cfg['name'] = name
                trans.append(LPC(**tr_cfg))
                #trans.append(LPC(opts.lpc_order, hop=opts.LPC_hop,
                #                 win=opts.LPC_win))
            elif 'fbank' in name:
                znorm = True
                tr_cfg['name'] = name
                trans.append(FBanks(**tr_cfg))
                #trans.append(FBanks(n_filters=opts.fbank_filters, 
                #                    n_fft=opts.nfft,
                #                    hop=opts.fbanks_hop,
                #                    win=opts.fbanks_win,
                #                    der_order=opts.fbanks_der_order))
            
            elif 'mfcc_librosa' in name:
                znorm = True
                tr_cfg['name'] = name
                trans.append(MFCC_librosa(**tr_cfg))
                #trans.append(MFCC_librosa(hop=opts.mfccs_librosa_hop, win=opts.mfccs_librosa_win, order=opts.mfccs_librosa_order, der_order=opts.mfccs_librosa_der_order, n_mels=opts.mfccs_librosa_n_mels, htk=opts.mfccs_librosa_htk))
            elif 'mfcc' in name:
                znorm = True
                tr_cfg['name'] = name
                trans.append(MFCC(**tr_cfg))
                #trans.append(MFCC(hop=opts.mfccs_hop, win=opts.mfccs_win, order=opts.mfccs_order, der_order=opts.mfccs_der_order))
            elif 'chroma' in name:
                znorm = True
                tr_cfg['name'] = name
                trans.append(Chroma(**tr_cfg))
            elif 'tempogram' in name:
                znorm = True
                tr_cfg['name'] = name
                trans.append(Tempogram(**tr_cfg))
            elif 'prosody' in name:
                znorm = True
                tr_cfg['name'] = name
                trans.append(Prosody(**tr_cfg))
                #trans.append(Prosody(hop=opts.prosody_hop, win=opts.prosody_win, der_order=opts.prosody_der_order))
            elif name == 'chunk' or name == 'cchunk':
                znorm = False
            elif 'kaldimfcc' in name:
                znorm = True
                tr_cfg['name'] = name
                trans.append(KaldiMFCC(**tr_cfg))
                #trans.append(KaldiMFCC(kaldi_root=opts.kaldi_root, hop=opts.kaldimfccs_hop, win=opts.kaldimfccs_win,num_mel_bins=opts.kaldimfccs_num_mel_bins,num_ceps=opts.kaldimfccs_num_ceps,der_order=opts.kaldimfccs_der_order))
            elif "kaldiplp" in name:
                znorm = True
                tr_cfg['name'] = name
                trans.append(KaldiPLP(**tr_cfg))
                #trans.append(KaldiPLP(kaldi_root=opts.kaldi_root, hop=opts.kaldiplp_hop, win=opts.kaldiplp_win))
            else:
                raise TypeError('Unrecognized module \"{}\"'
                                'whilst building transfromations'.format(name))
            keys.append(name)
    if znorm and stats is not None:
        trans.append(ZNorm(stats))
        keys.append('znorm')
    if trans_cache is None:
        trans = Compose(trans)
    else:
        trans = CachedCompose(trans, keys, trans_cache)
    return trans, collater_keys


class EndToEndImgDataset(Dataset):
    ''' End-to-end audio features to predict memorability '''
    def __init__(self, labels_df, config, split="train"):
        super().__init__()
        self.labels_df = labels_df
        self.split = split
        
        self.img_dir = config["path"]["img_dir"]
        
        if self.split != "test":
            # self.augmented_type_list = sorted(os.listdir(self.img_dir))[-1:]
            self.augmented_type_list = sorted(os.listdir(self.img_dir))[:]
        else:
            self.augmented_type_list = sorted(os.listdir(self.img_dir))[-1:]

        self.track_names = list(self.labels_df.track)
        self.scores = torch.tensor(self.labels_df.score)
        self.filename_to_score = dict(zip(self.track_names, self.scores))
        self.img_options = [[] for _ in range(len(self.labels_df))]
        
        self.mels_imgs = []
        self.transforms = transforms.Compose([
            transforms.Resize((config["model"]["image_size"], config["model"]["image_size"])),
            transforms.ToTensor()
        ])

        for track_name in tqdm(self.track_names):
            for augment_type in self.augmented_type_list:
                       
                img_path = os.path.join(self.img_dir, augment_type, "mels_"+track_name.replace(".wav", ".png"))        
                image = Image.open(img_path).convert('L') # convert to grayscale
                self.img_options[self.track_names.index(track_name)].append(self.transforms(image))


    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        img_options = self.img_options[index]
        img =  img_options[np.random.randint(0, len(img_options))]
        score = self.scores[index]
        return img, score

class PairEndToEndImgDataset(EndToEndImgDataset):
    ''' Hand crafted audio features to predict memorability (classification) '''
    def __init__(self, labels_df, config, pooling=False, split="train"):
        super().__init__(labels_df=labels_df, config=config, split=split)
        assert split in ["train", "valid"], "Split must be either train or valid"
        # get index combinations of wavs
        self.index_combinations = list(combinations([i for i in range(len(self.labels_df))], 2))
        self.img_combinations, self.score_combinations = [], []
        for index_pair in self.index_combinations:
            index_1, index_2 = index_pair
            self.img_combinations.append([self.img_options[index_1],
                                            self.img_options[index_2]])
            self.score_combinations.append([self.scores[index_1],
                                            self.scores[index_2]])


    def __len__(self):
        return len(self.index_combinations)

    def __getitem__(self, index):
        img_options_1, img_options_2 = self.img_combinations[index]
        img_1 = img_options_1[np.random.randint(0, len(img_options_1))]
        img_2 = img_options_2[np.random.randint(0, len(img_options_2))]
        score_1, score_2 = self.score_combinations[index]

        return img_1, img_2, score_1, score_2

def scale_minmax(X, min=0.0, max=1.0):
                X_std = (X - X.min()) / (X.max() - X.min())
                X_scaled = X_std * (max - min) + min
                return X_scaled

class AudioDataset(Dataset):
    ''' Audio features to predict memorability '''
    def __init__(self, labels_df, config, split="train"):
        super().__init__()
        self.sr = config["hparas"]["sample_rate"]
        self.labels_df = labels_df
        self.split = split
        self.track_names = list(self.labels_df.track)
        # self.scores = torch.tensor(self.labels_df.score)
        self.filename_to_score = dict(zip(self.track_names, self.labels_df.score))
        self.audio_root = config["path"]["audio_root"]
        self.audio_paths = []
        self.scores = []
        self.audios = []
        self.imgs = []
        if self.split != "test":
            self.audio_transforms = transforms.Compose([
                        VolChange(gain_range=config["augmentation"]["vol"]["gain_range"], \
                                prob=config["augmentation"]["vol"]["prob"]),
                        BandDrop(filt_files=config["augmentation"]["banddrop"]["ir_files"],\
                                data_root=config["augmentation"]["banddrop"]["data_root"], \
                                prob=config["augmentation"]["banddrop"]["prob"]),
                        Reverb(ir_files=[],\
                                ir_fmt=config["augmentation"]["reverb"]["ir_fmt"],\
                                data_root=config["augmentation"]["reverb"]["data_root"], \
                                prob=config["augmentation"]["reverb"]["prob"]),
                        SimpleAdditive(noises_dir=config["augmentation"]["add_noise"]["path"], \
                                        snr_levels=config["augmentation"]["add_noise"]["snr_options"], \
                                        prob=config["augmentation"]["add_noise"]["prob"]),
                        Fade(sample_rate=self.sr ,\
                            prob=config["augmentation"]["fade"]["prob"], \
                            fade_in_second=config["augmentation"]["fade"]["fade_in_sec"], \
                            fade_out_second=config["augmentation"]["fade"]["fade_out_sec"], \
                            fade_shape=config["augmentation"]["fade"]["fade_shape"]),

                        TimeStretch(rates=config["augmentation"]["time_stretch"]["speed_range"], \
                                    probability=config["augmentation"]["time_stretch"]["prob"]),
                        Melspectrogram(sample_rate=self.sr),
                    ])
        else:
            self.audio_transforms = transforms.Compose([
                        Melspectrogram(sample_rate=self.sr),
                    ])
        if isinstance (config["model"]["image_size"], list):
            size = tuple(config["model"]["image_size"]) # h, w
        else:
            size = tuple((config["model"]["image_size"], config["model"]["image_size"]))
        self.image_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

        if self.split != "test":
            self.audio_dirs = os.listdir(self.audio_root)
        else:
            self.audio_dirs = sorted(os.listdir(self.audio_root))[-1:]
        
        for audio_dir in tqdm(self.audio_dirs):
            for track_name in self.track_names:
                self.scores.append(torch.tensor(self.filename_to_score[track_name]))
                
                fname = os.path.join(self.audio_root, audio_dir, track_name)
                wav, rate = torchaudio.load(fname)
                # change sampling rate
                wav = torchaudio.transforms.Resample(rate, self.sr)(wav)
                wav = wav.numpy().squeeze()
                mels = self.audio_transforms(wav)
                # mels still to need to go through librosa.power_to_db, don't know why
                S_dB = librosa.power_to_db(mels, ref=np.max)
                S_dB = scale_minmax(np.flip(S_dB, axis=0), 0, 255)
                image = Image.fromarray(S_dB)
                
                self.imgs.append(self.image_transforms(image))
                
                
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        mels = self.imgs[index]
        score = self.scores[index]
        return mels, score

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AST_AudioDataset(Dataset):
    """ Modified from SSAST, ref: https://github.com/YuanGongND/ssast """
    def __init__(self, dataset_json_file, audio_conf, config):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        self.split = self.datapath.split('/')[-1].split('.')[0]

        if self.split != "test":
            self.sr = config["hparas"]["sample_rate"]
            self.audio_transforms = transforms.Compose([
                        VolChange(gain_range=config["augmentation"]["vol"]["gain_range"], \
                                prob=config["augmentation"]["vol"]["prob"]),
                        BandDrop(filt_files=config["augmentation"]["banddrop"]["ir_files"],\
                                data_root=config["augmentation"]["banddrop"]["data_root"], \
                                prob=config["augmentation"]["banddrop"]["prob"]),
                        Reverb(ir_files=[],\
                                ir_fmt=config["augmentation"]["reverb"]["ir_fmt"],\
                                data_root=config["augmentation"]["reverb"]["data_root"], \
                                prob=config["augmentation"]["reverb"]["prob"]),
                        SimpleAdditive(noises_dir=config["augmentation"]["add_noise"]["path"], \
                                        snr_levels=config["augmentation"]["add_noise"]["snr_options"], \
                                        prob=config["augmentation"]["add_noise"]["prob"]),
                        Fade(sample_rate=self.sr ,\
                            prob=config["augmentation"]["fade"]["prob"], \
                            fade_in_second=config["augmentation"]["fade"]["fade_in_sec"], \
                            fade_out_second=config["augmentation"]["fade"]["fade_out_sec"], \
                            fade_shape=config["augmentation"]["fade"]["fade_shape"]),

                        TimeStretch(rates=config["augmentation"]["time_stretch"]["speed_range"], \
                                    probability=config["augmentation"]["time_stretch"]["prob"]),
                    ])
        else:
            self.audio_transforms = transforms.Compose([])


        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = self.audio_transforms(waveform)  
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform).float().unsqueeze(0)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            raise Exception ("mixup is not supported for memorability regression task. Please set filename2=None")
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            raise Exception ("mixup is not supported for memorability regression task. Please set filename2=None")
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        """ between class learning, from: https://arxiv.org/pdf/1711.10282.pdf """
        if random.random() < self.mixup:
            raise Exception ("mixup is not supported for memorability regression task. Please set self.mixup to 0.0")
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            fbank, mix_lambda = self._wav2fbank(datum['wav'])

            label = torch.tensor(datum["labels"])

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label

    def __len__(self):
        return len(self.data)


class ReconstructionDataset(Dataset):
    ''' LSTM　hidden states to reconstruct mel-spectrograms ref: https://arxiv.org/pdf/1911.01102.pdf '''
    def __init__(self, labels_df, config, split="train"):

        super().__init__()
        self.downsampling_factor = config['model']['downsampling_factor']
        self.hidden_states_dir = config["path"]["hidden_states_dir"]
        self.mels_dir = config["path"]["mels_dir"]
        self.hidden_layer_num = config['experiment']['hidden_layer_num']
        self.mode = split
        self.augment_types = ["1_semitones", "2_semitones", "3_semitones", "4_semitones",
                        "5_semitones", "-1_semitones","-2_semitones", 
                        "-3_semitones", "-4_semitones", "-5_semitones", "original"]
        if self.mode == "test":
            self.augment_types = self.augment_types[-1]

        self.sequence_length = config["model"]["seq_len"]//self.downsampling_factor

            
        self.idx_to_filename = {idx: filename for idx, filename in enumerate(labels_df["track"])}

        self.hidden_states = []
        self.melspectrograms = []
        for augment_type in tqdm(self.augment_types, desc="defining dataset"):
            for audio_idx in range(len(self.idx_to_filename)):
                for layer_idx in range(self.hidden_layer_num):
                    hidden_states_path = os.path.join(self.hidden_states_dir,
                                    self.idx_to_filename[audio_idx],
                                    augment_type, str(layer_idx)+".npy")
                    hidden_states = np.load(hidden_states_path)
                    hidden_states = hidden_states[::self.downsampling_factor]               
                    self.hidden_states.append(hidden_states)
                
                melspetrogram_path = os.path.join(self.mels_dir, 
                                        augment_type, 
                                        "mels_{}.npy".format(self.idx_to_filename[audio_idx].replace(".wav", "")))

                melspectrogram = np.load(melspetrogram_path)
                # melspectrogram = melspectrogram[::self.downsampling_factor]
                self.melspectrograms.append(melspectrogram)       

    def __len__(self):
        return len(self.idx_to_filename)*len(self.augment_types)*self.hidden_layer_num*self.sequence_length

    def __getitem__(self, index):
        
        sequence_idx = index%self.sequence_length
        hidden_states = self.hidden_states[index//self.sequence_length][:,sequence_idx,:]
        hidden_states = torch.tensor(hidden_states)
        try:
            mels = torch.tensor(self.melspectrograms[index//(self.sequence_length*self.hidden_layer_num)][:, sequence_idx*self.downsampling_factor:(sequence_idx+1)*self.downsampling_factor], dtype=torch.double)
        except:
            raise Exception("index out of range")
        return hidden_states, mels
# CNN
def getData(mode):
    if mode == 'train':
        data = pd.read_csv('./music-regression/train.csv')
        audio_name = data.track
        score = data.score
        return np.squeeze(audio_name.values), np.squeeze(score.values)
    else:
        data = pd.read_csv('./music-regression/test.csv')
        audio_name = data.track
        return np.squeeze(audio_name.values)

class AudioUtil():
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      # Nothing to do
      return aud

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      resig = sig[:1, :]
    else:
      # Convert from mono to stereo by duplicating the first channel
      resig = torch.cat([sig, sig])

    return ((resig, sr))

  def resample(aud, newsr):
    sig, sr = aud

    if (sr == newsr):
      # Nothing to do
      return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
      retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])

    return ((resig, newsr))

  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)

  def time_shift(aud, shift_limit):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

class SoundDataset(Dataset):
    def __init__(self, labels_df, config, split):
        self.labels_df = labels_df
        self.track_names = list(self.labels_df.track)
        self.scores = []
        self.imgs = []
        self.weights = []
        self.filename_to_score = dict(zip(self.track_names, self.labels_df.score))
        self.audio_root = config["path"]["audio_root"]
        self.config = config
        self.split = split

        self.duration = self.config["hparas"]["duration"]
        self.sr = self.config["hparas"]["sample_rate"]
        self.channel = self.config["hparas"]["channel"]
        self.shift_pct = self.config["hparas"]["shift_pct"]

        
        if self.split != "test":
            self.audio_dirs = os.listdir(self.audio_root)
        else: # original
            self.audio_dirs = sorted(os.listdir(self.audio_root))[-1:]
        if self.split == "train":
            self.weights_distri = _prepare_weights(labels=self.labels_df.score, 
                                                    reweight="inverse", 
                                                    max_target=1, 
                                                    lds=True,
                                                    lds_ks=self.config["hparas"]["lds_ks"], 
                                                    lds_sigma=self.config["hparas"]["lds_sigma"],
                                                    bin_size=self.config["hparas"]["bin_size"])
        else:
            self.weights_distri = np.ones(len(self.labels_df.score)).astype(np.float32)


        for audio_dir in tqdm(self.audio_dirs):
            for track_name in self.track_names:
                self.scores.append(torch.tensor(self.filename_to_score[track_name]))
                audio_path = os.path.join(self.audio_root, audio_dir, track_name)
                aud = AudioUtil.open(audio_path)
                reaud = AudioUtil.resample(aud, self.sr)
                rechan = AudioUtil.rechannel(reaud, self.channel)

                # dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
                # shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
                sgram = AudioUtil.spectro_gram(rechan, n_mels=64, n_fft=1024, hop_len=None)
                if self.split != "test":
                    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
                    self.imgs.append(aug_sgram)
                else:
                    self.imgs.append(sgram)
                self.weights.append(self.weights_distri[self.track_names.index(track_name)])

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        img = self.imgs[index]
        score = self.scores[index]
        weight = self.weights[index]

        return img, score, weight

if __name__ == "__main__":
    pass
