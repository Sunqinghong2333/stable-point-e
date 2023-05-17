#!/usr/bin/env python3

"""Trains Karras et al. (2022) diffusion models."""

import argparse
import json
from ipdb import set_trace
import os

import torch
from torch import nn, optim
from torch import multiprocessing as mp
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
import numpy as np
import random

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.util import n_params
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.dataset.shapenet import ShapeNetCore
from point_e.util.point_cloud import PointCloud
from point_e.configs.config import load_config, make_sample_density
from point_e.util.common import get_linear_scheduler
from point_e.evals.metrics import *
from up.utils.general.log_helper import default_logger as logger
from up.utils.env.dist_helper import env, setup_distributed, finalize

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--save-root', type=str, default="./save_path",
                   help='save root for logger and ckpt')
    p.add_argument('--log-dir', type=str, default="./save_path",
                   help='save root for logger and ckpt')
    p.add_argument('--cond', type=str, default="texts",
                   help='condition type')
    args = p.parse_args()

    config = load_config(open(args.config))
    model_config = config['model']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    max_epoches = config['max_epoches']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if env.is_master():
        logger.info(os.environ['SLURM_NODELIST'])
        logger.info('creating base model...')
    name = model_config['name']
    base_model = model_from_config(MODEL_CONFIGS[name], device)
    base_model.to(device)
    logger.info('using ddp')
    base_model = DDP(base_model, device_ids=[env.local_rank], broadcast_buffers=False)
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[name])

    if env.is_master():
        logger.info('Parameters: {}'.format(n_params(base_model)))

    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(base_model.parameters(),
                          lr=opt_config['lr'],
                          betas=tuple(opt_config['betas']),
                          eps=opt_config['eps'],
                          weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(base_model.parameters(),
                        lr=opt_config['lr'],
                        momentum=opt_config.get('momentum', 0.),
                        nesterov=opt_config.get('nesterov', False),
                        weight_decay=opt_config.get('weight_decay', 0.))
    else:
        raise ValueError('Invalid optimizer type')

    if dataset_config["type"] == 'shapenet':
        train_set = ShapeNetCore(
            text_file=dataset_config["text_file"], 
            pcd_file=dataset_config["pcd_file"],
            num_points=model_config['num_points']
        )
    else:
        raise NotImplementedError

    if env.is_master():
        try:
            logger.info('Number of items in dataset: {}'.format(len(train_set)))
        except TypeError:
            pass


    if env.is_master():
        writer = SummaryWriter(args.log_dir)

    train_sampler = DistributedSampler(train_set)
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=model_config['batch_size'], num_workers=args.num_workers, persistent_workers=True, sampler=train_sampler, pin_memory=True, drop_last=True)
    logger.info('train_sampler: {}'.format(len(train_sampler)))
    logger.info('train_dl: {}'.format(len(train_dl)))

    max_steps = max_epoches * len(train_dl)
    if sched_config['type'] == 'linear':
        sched = get_linear_scheduler(
            opt,
            start_epoch=0,
            end_epoch=max_steps,
            start_lr=opt_config['lr'],
            end_lr= sched_config['min_lr']
            )
    else:
        assert False, 'sched type not support'
        
    sample_density = make_sample_density(model_config) # hyperparameter from original paper

    aux_channels = [] if '3channel' in model_config['name'] else ['R', 'G', 'B']
    sampler = PointCloudSampler(
        device=device,
        models=[base_model],
        diffusions=[base_diffusion],
        num_points=[model_config['num_points']],
        aux_channels=aux_channels,
        guidance_scale=[3.0],
        use_karras=[True],
        karras_steps=[64],
        sigma_min=[model_config['sigma_min']],
        sigma_max=[model_config['sigma_max']],
        s_churn=[3],
        model_kwargs_key_filter=[args.cond],
    )

    def save():
        filename = f'{args.save_root}/{step:08}.pth'
        logger.info(f'Saving to {filename}...')
        obj = {
            'model': sampler.models[0].state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'epoch': epoch,
            'step': step,
        }
        torch.save(obj, filename)
    
    step = 0
    for epoch in range(max_epoches):
        train_sampler.set_epoch(epoch)
        for batch in train_dl:
            cur_lr = sched.get_last_lr()[0]
            base_model.train() # train mode
            opt.zero_grad()
            reals = batch["pointcloud"]
            reals = reals.to(device)  

            if args.cond == "texts":
                cond = batch["desc"]
                losses = sampler.loss_texts(reals, cond, reals.shape[0])
            elif args.cond == "images":
                cond = batch["img"]
                cond = cond.to(device)
                losses = sampler.loss_images(reals, cond, reals.shape[0])
            else:
                raise NotImplementedError
            losses.backward()
            opt.step()
            sched.step()

            if env.is_master() and step % config['echo_every'] == 0:
                logger.info(f'Epoch: {epoch}, step: {step}, lr:{cur_lr:.6f}, losses: {losses.item():g}')
                writer.add_scalar('losses', losses.item(), global_step=step)

            if config['evaluate_every'] > 0 and step > 0 and step % config['evaluate_every'] == 0:
                test(step)
            if env.is_master() and step > 0 and step % config['save_every'] == 0:
                save()
            step += 1



if __name__ == '__main__':
    setup_distributed(20338, 'slurm', 'dist')
    main()
    finalize()
