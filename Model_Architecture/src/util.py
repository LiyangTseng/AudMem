# ref: https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch
from typing import Callable
from pathlib import Path
import os
import matplotlib.pyplot as plt
import math
import time
import itertools
from tqdm import tqdm
import torch.multiprocessing as mp
import json
from numpy.lib.format import open_memmap
import torch
import numpy as np
from numpy.linalg import norm
from torch import nn
import torch.nn.functional as F
from torch._six import inf
# import editdistance as ed
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
import matplotlib
matplotlib.use('Agg')


class Timer():
    ''' Timer for recording training time distribution. '''

    def __init__(self):
        self.prev_t = time.time()
        self.clear()

    def set(self):
        self.prev_t = time.time()

    def cnt(self, mode):
        self.time_table[mode] += time.time()-self.prev_t
        self.set()
        if mode == 'bw':
            self.click += 1

    def show(self):
        total_time = sum(self.time_table.values())
        self.time_table['avg'] = total_time/self.click
        self.time_table['rd'] = 100*self.time_table['rd']/total_time
        self.time_table['fw'] = 100*self.time_table['fw']/total_time
        self.time_table['bw'] = 100*self.time_table['bw']/total_time
        msg = '{avg:.3f} sec/step (rd {rd:.1f}% | fw {fw:.1f}% | bw {bw:.1f}%)'.format(
            **self.time_table)
        self.clear()
        return msg

    def clear(self):
        self.time_table = {'rd': 0, 'fw': 0, 'bw': 0}
        self.click = 0

# Reference : https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/e2e_asr.py#L168


def init_weights(module):
    # Exceptions
    if type(module) == nn.Embedding:
        module.weight.data.normal_(0, 1)
    else:
        for p in module.parameters():
            data = p.data
            if data.dim() == 1:
                # bias
                data.zero_()
            elif data.dim() == 2:
                # linear weight
                n = data.size(1)
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            elif data.dim() in [3, 4]:
                # conv weight
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            else:
                raise NotImplementedError


def init_gate(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)
    return bias

# Convert Tensor to Figure on tensorboard


def feat_to_fig(feat):
    # feat TxD tensor
    feat = feat.transpose(1, 0)
    data = _save_canvas(feat.numpy())
    return torch.FloatTensor(data), "HWC"


def _save_canvas(data, meta=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    if meta is None:
        ax.imshow(data, aspect="auto", origin="lower")
    else:
        ax.bar(meta[0], data[0], tick_label=meta[1], fc=(0, 0, 1, 0.5))
        ax.bar(meta[0], data[1], tick_label=meta[1], fc=(1, 0, 0, 0.5))
    fig.canvas.draw()
    # Note : torch tb add_image takes color as [0,1]
    data = np.array(fig.canvas.renderer._renderer)[:, :, :-1]/255.0
    plt.close(fig)
    return data

# Reference : https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python


def human_format(num):
    magnitude = 0
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '{:3.1f}{}'.format(num, [' ', 'K', 'M', 'G', 'T', 'P'][magnitude])


# def cal_er(tokenizer, pred, truth, mode='wer', ctc=False):
#     # Calculate error rate of a batch
#     if pred is None:
#         return np.nan
#     elif len(pred.shape) >= 3:
#         pred = pred.argmax(dim=-1)
#     er = []
#     for p, t in zip(pred, truth):
#         p = tokenizer.decode(p.tolist(), ignore_repeat=ctc)
#         t = tokenizer.decode(t.tolist())
#         if mode == 'wer':
#             p = p.split(' ')
#             t = t.split(' ')
#         er.append(float(ed.eval(p, t))/len(t))
#     return sum(er)/len(er)


def load_embedding(text_encoder, embedding_filepath):
    with open(embedding_filepath, "r") as f:
        vocab_size, embedding_size = [int(x)
                                      for x in f.readline().strip().split()]
        embeddings = np.zeros((text_encoder.vocab_size, embedding_size))

        unk_count = 0

        for line in f:
            vocab, emb = line.strip().split(" ", 1)
            # fasttext's <eos> is </s>
            if vocab == "</s>":
                vocab = "<eos>"

            if text_encoder.token_type == "subword":
                idx = text_encoder.spm.piece_to_id(vocab)
            else:
                # get rid of <eos>
                idx = text_encoder.encode(vocab)[0]

            if idx == text_encoder.unk_idx:
                unk_count += 1
                embeddings[idx] += np.asarray([float(x)
                                               for x in emb.split(" ")])
            else:
                # Suppose there is only one (w, v) pair in embedding file
                embeddings[idx] = np.asarray(
                    [float(x) for x in emb.split(" ")])

        # Average <unk> vector
        if unk_count != 0:
            embeddings[text_encoder.unk_idx] /= unk_count

        return embeddings


def freq_loss(pred, label, sample_rate, n_mels, loss, differential_loss, emphasize_linear_low, p=1):
    """
    Args:
        pred: model output
        label: target
        loss: `l1` or `mse`
        differential_loss: use differential loss or not, see here `https://arxiv.org/abs/1909.10302`
        emphasize_linear_low: emphasize the low-freq. part of linear spectrogram or not

    Return:
        loss
    """
    # ToDo : Tao
    # pred -> BxTxD predicted mel-spec or linear-spec
    # label-> same shape
    # return loss for loss.backward()
    if loss == 'l1':
        criterion = torch.nn.functional.l1_loss
    elif loss == 'mse':
        criterion = torch.nn.functional.mse_loss
    else:
        raise NotImplementedError

    cutoff_freq = 3000

    # Repeat for postnet
    _, chn, _, dim = pred.shape
    label = label.unsqueeze(1).repeat(1, chn, 1, 1)

    loss_all = criterion(p * pred, p * label)

    if dim != n_mels and emphasize_linear_low:
        # Linear
        n_priority_freq = int(dim * (cutoff_freq / (sample_rate/2)))
        pred_low = pred[:, :, :, :n_priority_freq]
        label_low = label[:, :, :, :n_priority_freq]
        loss_low = criterion(p * pred_low, p * label_low)
        #loss_low = torch.nn.functional.mse_loss(p * pred_low, p * label_low)
        loss_all = 0.5 * loss_all + 0.5 * loss_low

    if differential_loss:
        pred_diff = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        label_diff = label[:, :, 1:, :] - label[:, :, :-1, :]
        loss_all += 0.5 * criterion(p * pred_diff, p * label_diff)

    return loss_all


def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.

    We require ``max_length`` here instead of just computing it from the input ``sequence_lengths``
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return (sequence_lengths.unsqueeze(1) >= range_tensor).bool()


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm


def mp_progress_map(func, arg_iter, num_workers):
    rs = []
    pool = mp.Pool(processes=num_workers)
    for args in arg_iter:
        rs.append(pool.apply_async(func, args))
    pool.close()

    rets = []
    for r in tqdm(rs):
        rets.append(r.get())
    pool.join()
    return rets


def wave_to_feat_and_save_factory(wave_to_feat: Callable):
    global func

    def func(wavfile):
        feat = wave_to_feat(wavfile)
        wavfile = Path(wavfile)
        feat_file = wavfile.with_suffix('.pt')
        aug_feat_file = None
        aug_feat_len = None
        if isinstance(feat, tuple):
            feat, aug_feat = feat
            aug_feat_len = aug_feat.shape[0]
            aug_feat_file = wavfile.with_suffix('.aug.pt')
            torch.save(aug_feat, aug_feat_file)
        feat_len = feat.shape[0]
        torch.save(feat, feat_file)
        return feat_len, feat_file, aug_feat_len, aug_feat_file

    return func


def write_sliced_array(sequence, out_path, total_len,
                       dtype='float32', func=None):
    print('writing into sliced array...')
    tbar = tqdm(total=len(sequence))
    it = ((func(x), x) if func else (x, x) for x in sequence)
    x, path = next(it)
    #shape = (x.shape[0], total_len)
    shape = (total_len, x.shape[1])
    data = open_memmap(out_path, mode='w+', dtype=dtype, shape=shape)

    def put(entry, prev):
        now = prev + entry.shape[0]
        #data[:, prev:now] = entry
        data[prev:now, :] = entry
        tbar.update(1)
        return now
    prev = put(x, 0)
    for x, path in it:
        prev = put(x, prev)
        if isinstance(path, Path):
            os.remove(path)
    tbar.close()
    print('flushing...')
    data.flush()
    print('done.')


def cm_figure(y_true, y_pred, class_names, normalize=False, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        # Normalize the confusion matrix.
        # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    figure = plt.figure(figsize=(64, 64))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        if cm[i, j] != 0:
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def roc_score(spkr_emb, spkr_id):
    score_matrix, label_matrix, true_match, false_match = [], [], [], []
    for i in range(len(spkr_id)):
        for j in range(i):
            #score = np.dot(embeddings[i, :], embeddings[j, :])
            a = spkr_emb[i][:]
            b = spkr_emb[j][:]
            score = np.dot(a, b)/(norm(a)*norm(b))
            score_matrix.append(score)
            if (spkr_id[i] == spkr_id[j]):
                label_matrix.append(1)
                true_match.append(score)
            else:
                label_matrix.append(0)
                false_match.append(score)
    #print("There are %d positive and %d negative pairs" %(len(true_match), len(false_match)))

    fpr, tpr, _ = roc_curve(label_matrix, score_matrix)
    intersection = abs(1-tpr-fpr)
    eer = fpr[np.argmin(intersection)]
    dcf2 = np.min(100*(0.01*(1-tpr)+0.99*fpr))
    dcf3 = np.min(1000*(0.001*(1-tpr)+0.999*fpr))
    return eer, dcf2, dcf3

'=============================== from Pase ==============================='
''' source: https://github.com/santi-pdp/pase '''

class ContextualizedLoss(object):
    """ With a possible composition of r
        consecutive frames
    """

    def __init__(self, criterion, r=None):
        self.criterion = criterion
        self.r = r

    def contextualize_r(self, tensor):
        if self.r is None:
            return tensor
        assert isinstance(self.r, int), type(self.r)
        # ensure it is a 3-D tensor
        assert len(tensor.shape) == 3, tensor.shape
        # pad tensor in the edges with zeros
        pad_ = F.pad(tensor, (self.r // 2, self.r // 2))
        pt = []
        # Santi:
        # TODO: improve this with some proper transposition and stuff
        # rather than looping, at the expense of more memory I guess
        for t in range(pad_.size(2) - (self.r - 1)):
            chunk = pad_[:, :, t:t+self.r].contiguous().view(pad_.size(0),
                                                             -1).unsqueeze(2)
            pt.append(chunk)
        pt = torch.cat(pt, dim=2)
        return pt

    def __call__(self, pred, gtruth):
        gtruth_r = self.contextualize_r(gtruth)
        loss = self.criterion(pred, gtruth_r)
        return loss

def worker_parser(cfg_fname, batch_acum=1, device='cpu', do_losses=True,
                frontend=None):
    with open(cfg_fname, 'r') as cfg_f:
        cfg_list = json.load(cfg_f)
        if do_losses:
            # change loss section
            for type, cfg_all in cfg_list.items():

                for i, cfg in enumerate(cfg_all):
                    loss_name = cfg_all[i]['loss']
                    if hasattr(nn, loss_name):
                        # retrieve potential r frames parameter
                        r_frames = cfg_all[i].get('r', None)
                        # loss in nn Modules
                        cfg_all[i]['loss'] = ContextualizedLoss(getattr(nn, loss_name)(),
                                                                r=r_frames)
                    
            cfg_list[type] = cfg_all
        
        return cfg_list

def get_grad_norms(model, keys=[]):
    grads = {}
    for i, (k, param) in enumerate(dict(model.named_parameters()).items()):
        accept = False
        for key in keys:
            # match substring in collection of model keys
            if key in k:
                accept = True
                break
        if not accept:
            continue
        if param.grad is None:
            print('WARNING getting grads: {} param grad is None'.format(k))
            continue
        grads[k] = torch.norm(param.grad).cpu().item()
    return grads

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    ref: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

'=============================== from Deep Imbalanced Regression(DIR) ==============================='
''' source:https://github.com/YyzHarry/imbalanced-regression/'''

def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    
    if ks < 1:
        half_ks = ks
    else:
        half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

""" Label Distribution Smoothing(LDS), src: https://github.com/YyzHarry/imbalanced-regression/blob/main/agedb-dir/datasets.py#L55 """
def _prepare_weights(labels, reweight, max_target=121, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2, bin_size=1):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    value_dict = {x: 0 for x in range(int(max_target/bin_size))}
    
    for label in labels:
        value_dict[min((int((max_target-bin_size)/bin_size)), int((label-bin_size)/bin_size))] += 1
    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict[min((int((max_target-bin_size)/bin_size)), int((label-bin_size)/bin_size))] for label in labels]
    if not len(num_per_label) or reweight == 'none':
        return None
    print(f"Using re-weighting: [{reweight.upper()}]")

    if lds:
        # TODO: check if this size of kernel size, sigma is appropriate
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[min((int((max_target-bin_size)/bin_size)), int((label-bin_size)/bin_size))] for label in labels]

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights

""" Features Distribution Smoothing(FDS), src: https://github.com/YyzHarry/imbalanced-regression/blob/main/agedb-dir/fds.py """
class FDS(nn.Module):

    def __init__(self, feature_dim, bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma)
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(map(laplace, np.arange(-half_ks, half_ks + 1)))

        print(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).cuda()

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            print(f"Updated smoothed statistics on Epoch [{epoch}]!")

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                curr_feats = features[labels <= label]
            elif label == self.bucket_num - 1:
                curr_feats = features[labels >= label]
            else:
                curr_feats = features[labels == label]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[int(label - self.bucket_start)] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[int(label - self.bucket_start)]))
            factor = 0 if epoch == self.start_update else factor
            self.running_mean[int(label - self.bucket_start)] = \
                (1 - factor) * curr_mean + factor * self.running_mean[int(label - self.bucket_start)]
            self.running_var[int(label - self.bucket_start)] = \
                (1 - factor) * curr_var + factor * self.running_var[int(label - self.bucket_start)]

        print(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features

        labels = labels.squeeze(1)
        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                features[labels <= label] = calibrate_mean_var(
                    features[labels <= label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
            elif label == self.bucket_num - 1:
                features[labels >= label] = calibrate_mean_var(
                    features[labels >= label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
            else:
                features[labels == label] = calibrate_mean_var(
                    features[labels == label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
        return features
