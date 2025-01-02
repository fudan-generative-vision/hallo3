import os
import sys
import math
import random
import torch
import numpy as np
import json
import argparse
import warnings
from collections import OrderedDict, namedtuple
from sat import mpu
from sat.helpers import print_rank0, print_all
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Set, Tuple, Union
import itertools

from icecream import ic


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = '{:d}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d, 'mp_rank_{:02d}_model_states.pt'.format(mpu.get_model_parallel_rank()))


def get_checkpoint_tracker_filename(checkpoints_path, old_checkpoint=False):
    return os.path.join(checkpoints_path, 'latest')

def get_checkpoint_iteration(load_path):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        print_rank0('could not find the metadata file {} '.format(
            tracker_filename))
        raise ValueError('could not find the metadata file {}, please check --load'.format(
            tracker_filename))
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    return iteration, release, True


def load_checkpoint(model, args, load_path=None, prefix='', specific_iteration=None):
    """Load a model checkpoint."""
    if load_path is None:
        load_path = args.load

    # If model-only mode, set necessary args.
    if not hasattr(args, 'mode'):
        from copy import deepcopy
        args = deepcopy(args)
        args.mode = 'inference'

    iteration, release, success = get_checkpoint_iteration(load_path)
    if specific_iteration is not None:
        assert type(specific_iteration) == int and specific_iteration > 0
        print_rank0('Overriding checkpoint iteration to {}'.format(specific_iteration))
        iteration = specific_iteration
        
    if not success:
        return 0
    
    checkpoint_name = get_checkpoint_name(load_path, iteration, release)
    if mpu.get_data_parallel_rank() == 0:
            print_all('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))
            
    # load state_dict into CPU        
    sd = torch.load(checkpoint_name, map_location='cpu')

    # if given `prefix`, load a speficic prefix in the checkpoint, e.g. encoder
    new_sd = {'module':{}}
    for k in sd:
        if k != 'module':
            new_sd[k] = sd[k]
    for k in sd['module']:
        if k.startswith(prefix):
            new_sd['module'][k[len(prefix):]] = sd['module'][k]
    sd = new_sd
    
    if hasattr(model, 'module'):
        module = model.module
    else: # inference without deepspeed
        module = model

    # only load module, other hyperparameters are just for recording.
    missing_keys, unexpected_keys = module.load_state_dict(sd['module'], strict=False)
    
    if len(unexpected_keys) > 0:
        print_rank0(
            f'Will continue but found unexpected_keys! Check whether you are loading correct checkpoints: {unexpected_keys}.')
    if len(missing_keys) > 0:
        if args.mode == 'inference':
            if 'force_inference' in args and args.force_inference:
                print_rank0(f'Warning: Missing keys for inference: {missing_keys}.')
            else:
                raise ValueError(f'Missing keys for inference: {missing_keys}.\nIf you still want to inference anyway, pass --force_inference to args.')
        else: # new params
            if not args.force_train:
                assert all(name.find('mixins')>=0 or name.find('cross_attention')>=0 for name in missing_keys), missing_keys
                assert args.mode == 'finetune'
            # list all mixin names
            mixin_names = []
            for key_name in missing_keys:
                if key_name.find('mixins') < 0:
                    continue
                parts = key_name.split('.')
                mixin_name = parts[parts.index('mixins')+1]
                if mixin_name not in mixin_names:
                    mixin_names.append(mixin_name)
            module.reinit(mixin_names) # initialize mixins

    # Do not need this any more, because we create optimizer after load now.
    # if args.mode != 'inference' and args.deepspeed and args.fp16:
    #     model.optimizer.refresh_fp32_params() # restore fp32 weights from module

    # Iterations.
    if args.mode == 'finetune':
        iteration = 0
    elif args.mode == 'pretrain' and not args.no_load_rng: # rng states.
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank0('Unable to load optimizer from checkpoint {}, exiting. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the random '
                         'state.'.format(checkpoint_name))
            exit()
    elif args.mode == 'inference':
        module.eval()

    if mpu.get_data_parallel_rank() == 0:
        print_all('> successfully loaded {}'.format(checkpoint_name))
    del sd
    return iteration