import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import json
from src.transforms import *
import sys

class HandCraftedDataset(Dataset):
    ''' Hand crafted audio features to predict memorability (classification) '''
    def __init__(self, config, pooling=False, mode="train"):
        super().__init__()
        self.mode = mode
        self.features_dir = config["path"]["features_dir"]
        self.labels_dir = config["path"]["labels_dir"]
        
        if self.mode != "test":
            # self.augmented_type_list = sorted(os.listdir(self.features_dir))[-1:]
            self.augmented_type_list = sorted(os.listdir(self.features_dir))[:]
        else:
            self.augmented_type_list = sorted(os.listdir(self.features_dir))[-1:]
        self.pooling = pooling

        if self.mode == "train":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[:200]
        elif self.mode == "valid":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[200:220]
        elif self.mode == "test":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[220:]
        elif self.mode == "all":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))            
        else:
            raise Exception ("Invalid dataset mode")

        self.idx_to_filename = {idx: filename for idx, filename in enumerate(self.filename_memorability_df["track"])}
        self.idx_to_mem_score = {idx: score for idx, score in enumerate(self.filename_memorability_df["score"])}

        self.features_dict = config["features"]

        self.sequential_features = []
        self.non_sequential_features = []
        self.labels = []

        for augment_type in tqdm(self.augmented_type_list):
            for audio_idx in range(len(self.idx_to_filename)):
                
                sequeutial_features_list = []
                non_sequential_features_list = []
                
                for feature_type in self.features_dict:
                    for subfeatures in self.features_dict[feature_type]:
                        feature_file_path = os.path.join(self.features_dir, augment_type, feature_type, subfeatures,
                             "{}_{}".format(subfeatures, self.idx_to_filename[audio_idx].replace("wav", "npy")))   
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
                # store features
                self.sequential_features.append(torch.from_numpy(sequential_features))
                self.non_sequential_features.append(torch.from_numpy(np.stack(non_sequential_features_list)))
                self.labels.append(torch.tensor(self.idx_to_mem_score[audio_idx]))
        

    def __len__(self):
        return len(self.idx_to_filename)*len(self.augmented_type_list)       

    def __getitem__(self, index):
        if self.mode != "test":
            return self.sequential_features[index], self.non_sequential_features[index], self.labels[index]
        else:
            return self.sequential_features[index], self.non_sequential_features[index]

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
        with open(data_cfg_file, 'r') as data_cfg_f:
            self.data_cfg = json.load(data_cfg_f)
            wavs = self.data_cfg[split]['data']
            self.total_wav_dur = int(self.data_cfg[split]['total_wav_dur'])
            self.wavs = wavs
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
    ''' End-to-end audio features to predict memorability (classification) '''
    def __init__(self, config, mode="train"):
        super().__init__()
        self.mode = mode
        self.img_dir = config["path"]["img_dir"]
        self.labels_dir = config["path"]["labels_dir"]
        if self.mode != "test":
            # self.augmented_type_list = sorted(os.listdir(self.img_dir))[-1:]
            self.augmented_type_list = sorted(os.listdir(self.img_dir))[:]
        else:
            self.augmented_type_list = sorted(os.listdir(self.img_dir))[-1:]

        if self.mode == "train":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[:200]
        elif self.mode == "valid":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[200:220]
        elif self.mode == "test":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))[220:]
        elif self.mode == "all":
            self.filename_memorability_df = pd.read_csv(os.path.join(self.labels_dir, "track_memorability_scores_beta.csv"))            
        else:
            raise Exception ("Invalid dataset mode")

        self.idx_to_filename = {idx: filename for idx, filename in enumerate(self.filename_memorability_df["track"])}
        self.idx_to_mem_score = {idx: score for idx, score in enumerate(self.filename_memorability_df["score"])}
        
        self.labels = []
        self.mels_imgs = []
        self.transforms = transforms.Compose([
            transforms.Resize((config["model"]["image_size"], config["model"]["image_size"])),
            transforms.ToTensor()
        ])

        for augment_type in tqdm(self.augmented_type_list):
            for audio_idx in range(len(self.idx_to_filename)):         
                img_path = os.path.join(self.img_dir, augment_type, "mels_"+self.idx_to_filename[audio_idx].replace(".wav", ".png"))        
                image = Image.open(img_path).convert('L') # convert to grayscale
                self.mels_imgs.append(self.transforms(image))

                self.labels.append(torch.tensor(self.idx_to_mem_score[audio_idx]))


    def __len__(self):
        return len(self.idx_to_filename)*len(self.augmented_type_list)       

    def __getitem__(self, index):
        return self.mels_imgs[index], self.labels[index]

class ReconstructionDataset(Dataset):
    ''' LSTM　hidden states to reconstruct mel-spectrograms ref: https://arxiv.org/pdf/1911.01102.pdf '''
    def __init__(self, config, downsampling_factor=1, mode="train"):

        super().__init__()
        self.downsampling_factor = downsampling_factor
        self.hidden_states_dir = config["path"]["hidden_states_dir"]
        self.features_dir = config["path"]["features_dir"]
        self.hidden_layer_num = 4
        self.mode = mode
        # self.augmented_type_list = sorted(os.listdir(self.hidden_states_dir))[-1]
        if self.mode == "test":
            self.augmented_type_list = sorted(os.listdir(self.hidden_states_dir))[-1:]
        else:
            self.augmented_type_list = sorted(os.listdir(self.hidden_states_dir))[:]

        self.sequence_length = config["model"]["seq_len"]//self.downsampling_factor

        if self.mode == "train":
            filename_memorability_df = pd.read_csv("data/labels/track_memorability_scores_beta.csv")[:200]
        elif self.mode == "test":
            filename_memorability_df = pd.read_csv("data/labels/track_memorability_scores_beta.csv")[200:]
        else:
            raise Exception ("Invalid dataset mode")

            
        self.idx_to_filename = {idx: filename for idx, filename in enumerate(filename_memorability_df["track"])}

        self.hidden_states = []
        self.melspectrograms = []
        for augment_type in tqdm(self.augmented_type_list, desc="defining dataset"):
            for audio_idx in range(len(self.idx_to_filename)):
                for layer_idx in range(self.hidden_layer_num):
                    hidden_states_path = os.path.join(self.hidden_states_dir, augment_type,
                                    self.idx_to_filename[audio_idx].replace(".wav", ""), str(layer_idx)+".pt")
                    hidden_states = torch.load(hidden_states_path, map_location=torch.device('cpu'))
                    hidden_states = hidden_states[::self.downsampling_factor]               
                    self.hidden_states.append(hidden_states)
                
                if self.mode == "train":
                    melspetrogram_path = os.path.join(self.features_dir, 
                                            augment_type, "timbre", "melspectrogram", 
                                            "melspectrogram_{}.npy".format(self.idx_to_filename[audio_idx].replace(".wav", "")))

                    melspectrogram = np.load(melspetrogram_path)
                    # melspectrogram = melspectrogram[::self.downsampling_factor]
                    self.melspectrograms.append(melspectrogram)       

    def __len__(self):
        return len(self.idx_to_filename)*len(self.augmented_type_list)*self.hidden_layer_num*self.sequence_length

    def __getitem__(self, index):
        
        sequence_idx = index%self.sequence_length
        features = self.hidden_states[index//self.sequence_length][sequence_idx,:]
        features = features.detach() # this line is necessary for num_workers > 1
        if self.mode == "train":
            labels = torch.tensor(self.melspectrograms[index//(self.sequence_length*self.hidden_layer_num)][:, sequence_idx*self.downsampling_factor:(sequence_idx+1)*self.downsampling_factor], dtype=torch.double)
            return features, labels
        else:
            return features

if __name__ == "__main__":
    pass
