import torch
import torch.nn as nn
from ..frontend import WaveFe
from ..modules import *
import torch.nn.functional as F
import json
from torch.autograd import Function

def minion_maker(cfg):
    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = json.load(f)
    # print("=" * 50)
    # print("name", cfg["name"])
    # print("=" * 50)
    mtype = cfg.pop('type', 'mlp')
    if mtype == 'mlp':
        minion = MLPMinion(**cfg)
    elif mtype == 'decoder':
        minion = DecoderMinion(**cfg)
    else:
        raise TypeError('Unrecognized minion type {}'.format(mtype))
    return minion


class DecoderMinion(Model):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, 
                 dropout_time=0.0,
                 shuffle = False,
                 shuffle_depth = 7,
                 hidden_size=256,
                 hidden_layers=2,
                 fmaps=[256, 256, 128, 128, 128, 64, 64],
                 strides=[2, 2, 2, 2, 2, 5],
                 kwidths=[2, 2, 2, 2, 2, 5],
                 norm_type=None,
                 skip=False,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 name='DecoderMinion'):
        super().__init__(name=name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.dropout_time = dropout_time
        self.shuffle = shuffle
        self.shuffle_depth = shuffle_depth
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.fmaps = fmaps
        self.strides = strides
        self.kwidths = kwidths
        self.norm_type = norm_type
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys

        if keys is None:
            keys = [name]
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        # First go through deconvolving structure
        for (fmap, kw, stride) in zip(fmaps, kwidths, strides):
            block = GDeconv1DBlock(ninp, fmap, kw, stride,
                                   norm_type=norm_type)
            self.blocks.append(block)
            ninp = fmap

        for _ in range(hidden_layers):
            self.blocks.append(MLPBlock(ninp,
                                        hidden_size, dropout))
            ninp = hidden_size
        self.W = nn.Conv1d(hidden_size, num_outputs, 1)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        
        self.sg.apply(x, alpha)
        
        # The following part of the code drops out some time steps, but the worker should reconstruct all of them (i.e, the original signal)
        # This way we encourage learning features with a larger contextual information
        if self.dropout_time > 0:
            mask=(torch.FloatTensor(x.shape[0],x.shape[2]).to('cuda').uniform_() > self.dropout_time).float().unsqueeze(1)
            x=x*mask

        # The following function (when active) shuffles the time order of the input PASE features. Note that the shuffle has a certain depth (shuffle_depth). 
        # This allows shuffling features that are reasonably close, hopefully encouraging PASE to learn a longer context.
        if self.shuffle:
            x = torch.split(x, self.shuffle_depth, dim=2)
            shuffled_x=[]
            for elem in x:
                    r=torch.randperm(elem.shape[2])
                    shuffled_x.append(elem[:,:,r])

            x=torch.cat(shuffled_x,dim=2)

        h = x
        for bi, block in enumerate(self.blocks, start=1):
            h_ = h
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y

class MLPMinion(Model):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, dropout_time=0.0,hidden_size=256,
                 dropin=0.0,
                 hidden_layers=2,
                 context=1,
                 tie_context_weights=False,
                 skip=True,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 augment=False,
                 r=1, 
                 name='MLPMinion',
                 ratio_fixed=None, range_fixed=None, 
                 dropin_mode='std', drop_channels=False, emb_size=100):
        super().__init__(name=name)
        # Implemented with Conv1d layers to not
        # transpose anything in time, such that
        # frontend and minions are attached very simply
        self.num_inputs = num_inputs
        assert context % 2 != 0, context
        self.context = context
        self.tie_context_weights = tie_context_weights
        self.dropout = dropout
        self.dropout_time = dropout_time
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys
        if keys is None:
            keys = [name]
        # r frames predicted at once in the output
        self.r = r
        # multiplies number of output dims
        self.num_outputs = num_outputs * r
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        for hi in range(hidden_layers):
            self.blocks.append(MLPBlock(ninp,
                                        hidden_size,
                                        din=dropin,
                                        dout=dropout,
                                        context=context,
                                        tie_context_weights=tie_context_weights,
                                        emb_size=emb_size, 
                                        dropin_mode=dropin_mode,
                                        range_fixed=range_fixed,
                                        ratio_fixed=ratio_fixed,
                                        drop_channels=drop_channels))
            ninp = hidden_size
            # in case context has been assigned,
            # it is overwritten to 1
            context = 1
        self.W = nn.Conv1d(ninp, self.num_outputs, context,
                           padding=context//2)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        self.sg.apply(x, alpha)
        
        if self.dropout_time > 0 and self.context > 1:
            mask=(torch.FloatTensor(x.shape[0],x.shape[2]).to('cuda').uniform_() > self.dropout_time).float().unsqueeze(1)
            x=x*mask

        h = x
        for bi, block in enumerate(self.blocks, start=1):
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y

class ScaleGrad(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha

        return output, None


