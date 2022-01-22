import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.solver import BaseSolver
from src.util import worker_parser, get_grad_norms
from src.transforms import *
from models.pase_model import pase
from models.pase.WorkerScheduler.radam import *
from models.pase.WorkerScheduler.lr_scheduler import LR_Scheduler
from models.pase.WorkerScheduler.worker_scheduler import backprop_scheduler
from models.pase.Minions.minions import *
from src.dataset import build_dataset_providers, DictCollater
from src.util import human_format, get_grad_norm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import trange

CKPT_STEP = 10000

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.minions_cfg = worker_parser(self.config["path"]["net_cfg"])
        with open(self.config["path"]["fe_cfg"], 'r') as fe_cfg_f:
            self.fe_cfg = json.load(fe_cfg_f)

        self.CUDA = True if self.paras.gpu else False

        self.best_valid_loss = float('inf')

    def fetch_data(self, data):
        ''' Move data to device '''
        seq_feat, non_seq_feat, labeled_scores = data
        seq_feat, non_seq_feat, labeled_scores = seq_feat.to(self.device), non_seq_feat.to(self.device), labeled_scores.to(self.device)

        return seq_feat, non_seq_feat, labeled_scores

    def load_data(self):
        ''' Load data for training/validation '''
        dsets, collater_keys = build_dataset_providers(self.config, self.minions_cfg)
        self.train_set, self.valid_set = dsets
        self.verbose("Train data: found {} wav files".format(len(self.train_set.wavs)))
        self.verbose("Valid data: found {} wav files".format(len(self.valid_set.wavs)))
        self.verbose("Transform used: {}".format(self.train_set.transform))
        # Build collater, appending the keys from the loaded transforms to the
        # existing default ones
        collater = DictCollater()
        collater.batching_keys.extend(collater_keys)
        self.train_loader = DataLoader(self.train_set, batch_size=self.config["experiment"]["batch_size"],
                            shuffle=True, collate_fn=collater,
                            num_workers=self.config["experiment"]["num_workers"],drop_last=True,
                            pin_memory=self.CUDA)
        # Compute estimation of bpe. As we sample chunks randomly, we
        # should say that an epoch happened after seeing at least as many
        # chunks as total_train_wav_dur // chunk_size
        bpe = (self.train_set.total_wav_dur // self.config["dataset"]["chunk_size"]) // self.config["experiment"]["batch_size"]
        self.verbose("Train data length: {}".format(self.train_set.total_wav_dur/16000/3600.0))
        self.config["dataset"]["bpe"] = bpe
        if self.config["dataset"]["do_eval"]:
            assert self.valid_set is not None, (
                "Asked to do validation, but failed to build validation set"
            )
            self.valid_loader = DataLoader(self.valid_set, batch_size=self.config["experiment"]["batch_size"],
                                    shuffle=True, collate_fn=DictCollater(),
                                    num_workers=self.config["experiment"]["num_workers"],drop_last=True,
                                    pin_memory=self.CUDA)
            va_bpe = (self.valid_set.total_wav_dur // self.config["dataset"]["chunk_size"]) // self.config["experiment"]["batch_size"]
            self.config["dataset"]["va_bpe"] = va_bpe
        else:
            self.valid_loader = None
        # fastet lr to MI
        #self.config.min_lrs = {'mi':0.001}
        
        data_msg = ('I/O spec.  | Self-supervised workers = {}\t'
                .format(collater_keys))

        self.verbose(data_msg)

    def set_model(self):
        ''' Setup pase model and self.configimizer '''

        regr_lst = []

        if len(regr_lst) == 0 and 'regr' in self.minions_cfg:
            regr_lst = [worker['name'] for worker in self.minions_cfg['regr']]

        self.verbose('Regression minions: {}'.format(regr_lst))

        self.verbose("Pase config: {}".format(self.fe_cfg))
        self.model = pase(frontend=None,
                            frontend_cfg=self.fe_cfg,
                            minions_cfg=self.minions_cfg,
                            cls_lst=[], regr_lst=regr_lst,
                            pretrained_ckpt=None,
                            name='Pase_base').to(self.device)
        self.verbose(self.model.create_msg())

        # init param
        self.epoch = self.config["hparas"]['max_epoch']
        self.bsize = self.config["experiment"]['batch_size']
        self.log_freq = self.config["experiment"]['log_freq']
        self.bpe = self.config["dataset"]["bpe"]
        self.va_bpe = self.config["dataset"]["va_bpe"]
        self.savers = []
        self.fronted_cfg = self.fe_cfg
        self.cfg = self.config



        if self.config["hparas"]['fe_opt'].lower() == 'radam':
            self.frontend_optim = RAdam(self.model.frontend.parameters(),
                                        lr=self.config["hparas"]['fe_lr'])
        else:
            # init front end optim
            self.frontend_optim = getattr(optim, self.config["hparas"]['fe_opt'])(self.model.frontend.parameters(),
                                                  lr=self.config["hparas"]['fe_lr'])
        self.fe_scheduler = LR_Scheduler(self.config["hparas"]["lr_mode"], lr_step=self.config["hparas"]['lrdec_step'], optim_name="frontend", base_lr=self.config["hparas"]['fe_lr'],
                                    num_epochs=self.epoch,
                                    iters_per_epoch=self.bpe)
        self.verbose("Lr_Scheduler: using {} for {}".format(self.config["hparas"]["lr_mode"], "frontend"))
            

        self.savers.append(Saver(self.model.frontend, self.logdir,
                        max_ckpts=self.config["hparas"]['max_ckpts'],
                        optimizer=self.frontend_optim, prefix='PASE-'))

        # init workers optim

        self.cls_optim, self.cls_scheduler = {}, {} # currently useless
        self.regr_optim = {}
        self.regr_scheduler = {}
        for worker in self.model.regression_workers:
            min_opt = self.config["hparas"]['min_opt']
            min_lr = self.config["hparas"]['min_lr']
            # could be a regularizer minion
            if min_opt.lower() == 'radam':
                self.regr_optim[worker.name] = RAdam(worker.parameters(),
                                                     lr=min_lr)
            else:
                self.regr_optim[worker.name] = getattr(optim, min_opt)(worker.parameters(),
                                                                       lr=min_lr)
            worker_scheduler = LR_Scheduler(self.config["hparas"]["lr_mode"], lr_step=self.config["hparas"]['lrdec_step'], optim_name=worker.name, base_lr=min_lr,
                                            num_epochs=self.epoch,
                                            iters_per_epoch=self.bpe)
            self.verbose("Lr_Scheduler: using {} for {}".format(self.config["hparas"]["lr_mode"], worker.name))
            self.regr_scheduler[worker.name] = worker_scheduler

            self.savers.append(Saver(worker, self.logdir, max_ckpts=self.config["hparas"]['max_ckpts'],
                                     optimizer=self.regr_optim[worker.name],
                                     prefix='M-{}-'.format(worker.name)))

        self.epoch_beg = 0


        # init tensorboard writer
        self.verbose("Use tenoserboard: {}".format(self.config["experiment"]["tensorboard"]))

        # init backprop scheduler
        assert self.config["hparas"]["backprop_mode"] is not None
        self.backprop = backprop_scheduler(self.model, mode=self.config["hparas"]["backprop_mode"])
        self.alphaSG = 1

        self.worker_drop_rate = None
        self.delta = None
        self.temp = None
        self.alpha = None

        # auto supervise task evaluation
        if self.config["experiment"]['sup_exec'] is not None:
            aux_save_path = os.path.join(self.logdir,
                                             'sup_aux')
            if not os.path.exists(aux_save_path):
                os.makedirs(aux_save_path)
            self.aux_sup = AuxiliarSuperviser(self.config["experiment"]['sup_exec']['sup_exec'], aux_save_path)
        self.sup_freq = self.config["experiment"]['sup_freq']

    def backward(self, preds, labels, batch):
        '''
        Standard backward step with self.timer and debugger
        Arguments
            loss - the loss to perform loss.backward()
        '''
        self.timer.set()
        losses, self.alphaSG = self.backprop(preds,
                                            labels,
                                            self.cls_optim,
                                            self.regr_optim,
                                            self.frontend_optim,
                                            device=self.device,
                                            dropout_rate=self.worker_drop_rate,
                                            delta=self.delta,
                                            temperture=self.temp,
                                            alpha=self.alpha,
                                            batch = batch)
        
        return losses, self.alphaSG

    def load_ckpt(self, cont=True):
        ''' Load ckpt if --load self.configion is specified '''
        if self.paras.load:
            # Load weights
            ckpt = torch.load(
                self.paras.load, map_location=self.device if self.mode == 'train' else 'cpu')
            
            self.asr.load_state_dict(ckpt['model'])
            if self.emb_decoder is not None:
                self.emb_decoder.load_state_dict(ckpt['emb_decoder'])

            # Load task-dependent items
            if self.mode == 'train':
                if cont:
                    self.tts.load_state_dict(ckpt['tts'])
                    self.step = ckpt['global_step']
                    self.self.configimizer.load_self.config_state_dict(ckpt['self.configimizer'])
                    self.verbose('Load ckpt from {}, restarting at step {}'.format(
                        self.paras.load, self.step))
            else:
                raise NotImplementedError

    def save_checkpoint(self, f_name, pretext_losses):
        ''''
        Ckpt saver
            f_name - <str> the name phnof ckpt file (w/o prefix) to store, overwrite if existed
            score  - <float> The value of metric used to evaluate model
        '''
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "model": self.model.frontend.state_dict(),
            "frontend_optimizer": self.frontend_optim.state_dict(),
            "global_epoch": self.epoch
        }

        for worker_name, loss in pretext_losses.items():
            full_dict[worker_name+"_loss"] = float(loss)


        torch.save(full_dict, ckpt_path)
        self.verbose("Saved checkpoint (step = {}, {} = {:.2f}) and status @ {}".
                     format(human_format(self.epoch), "total loss", pretext_losses["total"], ckpt_path))

    def train_logger(self, preds, labels, losses, epoch, bidx, lrs, pbar):
        step = epoch * self.bpe + bidx
        pbar.write("=" * 50)
        pbar.write('Batch {}/{} (Epoch {}) step: {}:'.format(bidx, self.bpe, epoch, step))

        for name, loss in losses.items():
            if name == "total":
                pbar.write('%s, learning rate = %.8f, loss = %.4f' % ("total", lrs['frontend'], loss))
            else:
                pbar.write('%s, learning rate = %.8f, loss = %.4f' % (name, lrs[name], loss))


            self.log.add_scalar('train/{}_loss'.format(name),
                                loss.item(),
                                global_step=step)

            if name != "total":
                self.log.add_histogram('train/{}'.format(name),
                                        preds[name].data,
                                        bins='sturges',
                                        global_step=step)

                self.log.add_histogram('train/gtruth_{}'.format(name),
                                        labels[name].data,
                                        bins='sturges',
                                        global_step=step)

        grads = get_grad_norms(self.model)
        for kgrad, vgrad in grads.items():
            self.log.add_scalar('train/GRAD/{}'.format(kgrad),
                              vgrad, global_step=step)

    def eval_logger(self, running_loss, epoch, pbar):
        
        for name, loss in running_loss.items():
            loss = np.mean(loss)
            pbar.write("avg loss {}: {}".format(name, loss))

            self.log.add_scalar('eval/{}_loss'.format(name),
                                    loss,
                                    global_step=epoch)


    def exec(self):
        ''' Self-Supervised Pre-Training Pase '''
        
        self.verbose("Training pase...")
        self.verbose('Loss schedule policy: {}'.format(self.backprop.mode))
        self.timer.set()

        if self.config["experiment"]["ckpt_continue"]:
            # TODO: copy from pase
            # self.resume_training(device)
            pass
        else:
            self.epoch_beg = 0

        for epoch in range(self.epoch_beg, self.epoch):

            self.model.train()
            iterator = iter(self.train_loader)

            with trange(1, self.bpe + 1) as pbar:
                for bidx in pbar:
                    pbar.set_description("Epoch {}/{}".format(epoch, self.epoch))
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(self.train_loader)
                        batch = next(iterator)

                    # inference
                    h, chunk, preds, labels = self.model.forward(batch, self.alphaSG, self.device)
                    losses, self.alphaSG = self.backward(preds, labels, batch)
                
                    if bidx % self.log_freq == 0 or bidx >= self.bpe:
                        # decrease learning rate
                        lrs = {}
                        lrs["frontend"] = self.fe_scheduler(self.frontend_optim, bidx, epoch, losses["total"].item())

                        for name, scheduler in self.cls_scheduler.items():
                            lrs[name] = scheduler(self.cls_optim[name], bidx, epoch, losses[name].item())

                        for name, scheduler in self.regr_scheduler.items():
                            lrs[name] = scheduler(self.regr_optim[name], bidx, epoch, losses[name].item())

                        for k in losses.keys():
                            if k not in lrs:
                                lrs[k] = 0

                        self.train_logger(preds, labels, losses, epoch, bidx, lrs, pbar)


                    self.step += 1

            self.validate(epoch=epoch)

            self.save_checkpoint(f_name='FE_e{}.ckpt'.format(epoch), pretext_losses=losses)
            for saver in self.savers:
                saver.save(saver.prefix[:-1], epoch * self.bpe + bidx)


        self.log.close()
        
    def validate(self, epoch):
        # Eval mode
        self.model.eval()
        with torch.no_grad():
            self.verbose("Evaluate pase...")
            running_loss = {}
            iterator = iter(self.valid_loader)

            with trange(1, self.va_bpe + 1) as pbar:
                for bidx in pbar:
                    pbar.set_description("Eval: {}/{}".format(bidx, self.va_bpe+1))
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(self.valid_loader)
                        batch = next(iterator)

                    # inference
                    h, chunk, preds, labels = self.model.forward(batch, device=self.device)

                    # calculate losses
                    tot_loss = torch.tensor([0.]).to(self.device)
                    losses = {}
                    for worker in self.model.classification_workers:
                        loss = worker.loss(preds[worker.name], labels[worker.name])
                        losses[worker.name] = loss
                        tot_loss += loss
                        if worker.name not in running_loss:
                            running_loss[worker.name] = [loss.item()]
                        else:
                            running_loss[worker.name].append(loss.item())

                    for worker in self.model.regression_workers:
                        loss = worker.loss(preds[worker.name], labels[worker.name])
                        losses[worker.name] = loss
                        tot_loss += loss
                        if worker.name not in running_loss:
                            running_loss[worker.name] = [loss.item()]
                        else:
                            running_loss[worker.name].append(loss.item())
                    if 'total' not in running_loss:
                        running_loss["total"] = [tot_loss.item()]
                    else:
                        running_loss["total"].append(tot_loss.item())

                    if bidx % self.log_freq == 0 or bidx >= self.bpe:
                        pbar.write('-' * 50)
                        pbar.write('EVAL Batch {}/{} (Epoch {}):'.format(bidx,
                                                                    self.va_bpe,
                                                                    epoch))
                        for name, loss in losses.items():
                            pbar.write('{} loss: {:.3f}'
                                  ''.format(name, loss.item()))
            self.eval_logger(running_loss, epoch, pbar)


class AuxiliarSuperviser(object):

    def __init__(self, cmd_file, save_path='.'):
        self.cmd_file = cmd_file
        with open(cmd_file, 'r') as cmd_f:
            self.cmd = [l.rstrip() for l in cmd_f]
        self.save_path = save_path

    def __call__(self, iteration, ckpt_path, cfg_path):
        assert isinstance(iteration, int)
        assert isinstance(ckpt_path, str)
        assert isinstance(cfg_path, str)
        for cmd in self.cmd:
            sub_cmd = cmd.replace('$model', ckpt_path)
            sub_cmd = sub_cmd.replace('$iteration', str(iteration))
            sub_cmd = sub_cmd.replace('$cfg', cfg_path)
            sub_cmd = sub_cmd.replace('$save_path', self.logdir)
            print('Executing async command: ', sub_cmd)
            #shsub = shlex.split(sub_cmd)
            #print(shsub)
            p = subprocess.Popen(sub_cmd,
                                shell=True)

