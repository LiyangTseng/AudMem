# ref: https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch
import os
import sys
import abc
import math
import yaml
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from src.option import default_hparas
from src.util import human_format, Timer

class BaseSolver():
    ''' 
    Prototype Solver for all kinds of tasks
    Arguments
        config - yaml-styled config
        paras  - argparse outcome
    '''
    def __init__(self, config, paras, mode):
        # General Settings
        self.config = config
        self.paras = paras
        self.mode = mode
        for k,v in default_hparas.items():
            setattr(self,k,v)

        self.device = torch.device('cuda') if self.paras.gpu and torch.cuda.is_available() else torch.device('cpu')
        
        if mode == 'train':
            self.exp_name = paras.name
            if self.exp_name is None:
                now_tuple = time.localtime(time.time())
                self.exp_name = '%02d-%02d-%02d_%02d:%02d'%(now_tuple[0]%100,now_tuple[1],now_tuple[2],now_tuple[3],now_tuple[4])
            
            # Filepath setup
            os.makedirs(paras.ckpdir, exist_ok=True)
            self.ckpdir = os.path.join(paras.ckpdir, paras.model, self.exp_name)
            os.makedirs(self.ckpdir, exist_ok=True)

            # Logger settings
            self.logdir = os.path.join(paras.logdir, paras.model, self.exp_name)
            self.log = SummaryWriter(self.logdir)
            self.timer = Timer()

            # Hyperparameters
            self.step = 0
            self.valid_step = config['hparas']['valid_step']
            self.max_step = config['hparas']['max_step']
            self.epoch = 1
            self.max_epoch = config['hparas']['max_epoch']
            if self.paras.patience > 0:
                from utils.early_stopping_pytorch.pytorchtools import EarlyStopping
                self.early_stopping = EarlyStopping(patience=self.paras.patience, verbose=True)

            
            self.verbose('Exp. name : {}'.format(self.exp_name))
            self.verbose('Loading data... large corpus may took a while.')
            
        elif mode == 'test':
            # Output path
            self.outdir = self.paras.outdir
            os.makedirs(self.outdir, exist_ok=True)
            self.verbose('Evaluating result of tr. config @ config/{}.yaml'.format(self.paras.model)) 

    def backward(self, loss):
        '''
        Standard backward step with self.timer and debugger
        Arguments
            loss - the loss to perform loss.backward()
        '''
        self.timer.set()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.GRAD_CLIP)
        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step '+str(self.step))
        else:
            self.optimizer.step()
        self.timer.cnt('bw')
        return grad_norm

    def load_ckpt(self, cont=True):
        ''' Load ckpt if --load option is specified '''
        if self.paras.load:
            # Load weights
            ckpt = torch.load(self.paras.load, map_location=self.device if self.mode=='train' else 'cpu')
            if type(ckpt) == dict:
                self.model.load_state_dict(ckpt['model'])
            else:
                self.model.load_state_dict(ckpt)

            if self.mode == 'train':
                if cont:
                    self.step = ckpt['global_step']
                    self.optimizer.load_opt_state_dict(ckpt['optimizer'])
                    self.verbose('Load ckpt from {}, restarting at step {}'.format(self.paras.load,self.step))
            else:
                if type(ckpt) == dict:
                    for k,v in ckpt.items():
                        if type(v) is float:
                            metric, score = k,v
                    self.model.eval()
                    self.verbose('Evaluation target = {} (recorded {} = {:.2f} %)'.format(self.paras.load,metric,score))
                else:
                    self.model.eval()

    def verbose(self,msg):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            if type(msg)==list:
                for m in msg:
                    print('[INFO]',m.ljust(100))
            else:
                print('[INFO]',msg.ljust(100))

    def progress(self,msg):
        ''' Verbose function for updating progress on stdout (do not include newline) '''
        if self.paras.verbose:
            sys.stdout.write("\033[K") # Clear line
            print('[{}] {}'.format(human_format(self.step),msg),end='\r')
    
    def write_log(self,log_name,log_value):
        '''
        Write log to TensorBoard
            log_name  - <str> Name of tensorboard variable 
            log_value - <dict>/<array> Value of variable (e.g. dict of losses), passed if value = None
        '''
        if type(log_value) is dict:
            log_value = {key:val for key, val in log_value.items() if (val is not None and not math.isnan(val))}
        if log_value is None:
            pass
        elif len(log_value)>0:
            # ToDo : support all types of input
            if issubclass(type(log_name), torch.nn.Module):
                self.log.add_graph(log_name, log_value)
            elif 'distri' in log_name:
                self.log.add_histogram(log_name, log_value, self.epoch)
            elif 'align' in log_name or 'spec' in log_name or 'hist' in log_name:
                img, form = log_value
                self.log.add_image(log_name,img, global_step=self.epoch, dataformats=form)
            elif 'code' in log_name:
                self.log.add_embedding(log_value[0], metadata=log_value[1], tag=log_name, global_step=self.epoch)
            elif 'wave' in log_name:
                signal, sr = log_value
                self.log.add_audio(log_name, torch.FloatTensor(signal).unsqueeze(0), self.epoch, sr)
            elif 'text' in log_name or 'hyp' in log_name:
                self.log.add_text(log_name, log_value, self.epoch)
            else:
                self.log.add_scalars(log_name,log_value,self.epoch)

    def save_checkpoint(self, f_name, metric, score):
        '''' 
        Ckpt saver
            f_name - <str> the name phnof ckpt file (w/o prefix) to store, overwrite if existed
            score  - <float> The value of metric used to evaluate model
        '''
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.get_opt_state_dict(),
            # "global_step": self.step,
            "global_epoch": self.epoch,
            metric: score
        }

        torch.save(full_dict, ckpt_path)
        self.verbose("Saved checkpoint (step = {}, {} = {:.2f}) and status @ {}".\
                                       format(human_format(self.epoch),metric,score,ckpt_path))

    def enable_apex(self):
        if self.amp:
            # Enable mixed precision computation (ToDo: Save/Load amp)
            from apex import amp
            self.amp_lib = amp
            self.verbose("AMP enabled (check https://github.com/NVIDIA/apex for more details).")
            self.model, self.optimizer.opt = self.amp_lib.initialize(self.model, self.optimizer.opt, opt_level='O1')


    # ----------------------------------- Abtract Methods ------------------------------------------ #
    @abc.abstractmethod
    def load_data(self):
        '''
        Called by main to load all data
        After this call, data related attributes should be setup (e.g. self.tr_set, self.dev_set)
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def set_model(self):
        '''
        Called by main to set models
        After this call, model related attributes should be setup (e.g. self.l2_loss)
        The followings MUST be setup
            - self.model (torch.nn.Module)
            - self.optimizer (src.Optimizer),
                init. w/ self.optimizer = src.Optimizer(self.model.parameters(),**self.config['hparas'])
        Loading pre-trained model should also be performed here 
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def exec(self):
        '''
        Called by main to execute training/inference
        '''
        raise NotImplementedError


