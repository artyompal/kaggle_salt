# --- HYPERPARAMETERS

MODEL_NAME = 'ternausv2_pretrained'
RESIZE = False
NORMALIZATION = 'default'
LOAD_WEIGHTS = True
BATCH_SIZE = 30
OPT = 'adam'
LR = 1e-5
EPOCHS = 100
LOSS = 'bce_with_dice'
METRICS = 'loss'

def merge_options(custom_options={}):
    options = {
        'augment': { ('image', 'mask'): [
            #{'type': 'PadIfNeeded', 'min_height': 128, 'min_width': 202},
            #{'type': 'RandomCrop', 'height': 128, 'width': 128},
            #{'type': 'HorizontalFlip'},
            #{'type': 'Blur'},
        ] },
	'norm_type': NORMALIZATION,
        'batch_size': BATCH_SIZE,
        'optimizer': (OPT, { 'lr': LR }),
        'epochs': EPOCHS,
        'loss': LOSS,
        'metrics': METRICS.split(',')
    }
    for k, v in custom_options.items():
        options[k] = v
    return copy.deepcopy(options)

# --- END HYPERPARAMETERS

# Custom runner for ternaus net arch
# The difference with basic unet can be in pre-post processing need

import torch

# Lenin straight from repo
import sys
sys.path.insert(0, "/home/saint/gazay/berloga-dl/lenin")
import lenin
from lenin import train, test
from lenin.datasets.salt import Dataset

# Ternaus net from our repo
sys.path.insert(0, "/home/saint/gazay/berloga-dl/salt")
from src.models import MODELS
from src.utils import *

# Torch and torchbearer
import torch
from torch import nn
from torch.nn import functional as F
from torchbearer.callbacks.checkpointers import Best
from torchbearer.callbacks.csv_logger import CSVLogger
import torchbearer.callbacks.torch_scheduler as schedulers
from torch.autograd import Variable

# stdlib
import json
import copy
import string
import random
from collections import OrderedDict
import numpy as np
import pandas as pd
from imageio import imread

# CONSTANTS
root = "/home/saint/"
DIRECTORY = root + "datasets/competitions/tgs-salt-identification-challenge"
SAVE_PATH = root + "models/salt/"
SUBMISSIONS_PATH = root + "submissions/salt/"
WEIGHTS_PATH = root + "weights/ternaus_net_v2_deepglobe_buildings.pt"


# Pre/postprocessors
from skimage.transform import resize
import albumentations as aug
pad = aug.PadIfNeeded(min_height=128, min_width=128)
crop = aug.CenterCrop(height=101, width=101)
def upsample(img):
    if RESIZE:
        img = resize(img, (128, 128), mode='constant', preserve_range=True)
    else:
        img = pad(image=img)['image']
    return img

def downsample(img):
    if RESIZE:
        img = resize(img, (101, 101), mode='constant', preserve_range=True)
    else:
        img = crop(image=img)['image']
    return img

imgs_mean = 0.469914
imgs_std = 0.163075
cumsums_mean = 0.23724
cumsums_std = 0.15206
depths_mean = 0.52639
depths_std = 0.214675
NORMS = {
    # First channel – color. Second – cumsum. Third – depth with scaling
    'three': aug.Normalize(mean=(imgs_mean,cumsums_mean,depths_mean), std=(imgs_std,cumsums_std,depths_std)),
    # Custom normalization for all three similar channels
    'custom': aug.Normalize(mean=(imgs_mean,imgs_mean,imgs_mean), std=(imgs_std,imgs_std,imgs_std)),
    # Standard normalization for all three similar channels =/
    'default': aug.Normalize(mean=(0.485,0.485,0.485), std=(0.229,0.229,0.229))
}
norm = NORMS[NORMALIZATION]
def img_preprocess(img):
    img = upsample(img)
    img = norm(image=img)['image']
    return img

def img_postprocess(img):
    img = np.transpose(img, (2, 0, 1)).astype('float32')
    return img

def mask_preprocess(mask):
    mask = upsample(mask)
    return mask

def mask_postprocess(mask):
    ones = np.expand_dims(mask, 2) > 0
    zeros = ones != True
    mask = np.concatenate((ones, zeros), axis=2)
    mask = np.transpose(mask, (2, 0, 1)).astype('float32')
    return mask

class SaltDataset(Dataset):
    def __init__(self, root):
        super(SaltDataset, self).__init__(root)

    def image(self, record, augmentor=lambda x: x):
        img = imread(self.path(record, 'image'))
        if NORMALIZATION == 'three':
            # Add cumsum of img:
            img[:, :, 1] = self.cumsum_channel(img[:, :, 0])
            # Add depth of img
            img[:, :, 2] = self.depths_channel(self.depth(record), img[:,:,0].shape)
        img = augmentor(img)
        return img

    def depths_channel(self, depth, dim):
        depths_scaling = 0.1
        depths_max = 959. + 128. * depths_scaling # 128 – height of scaled image

        depths_arr = np.cumsum(np.ones(dim) * depths_scaling, axis=0) + depth
        depths_arr = (depths_arr / depths_max) * 255.
        return depths_arr

    def cumsum_channel(self, channel):
        cumsums_max = 25755.

        cumsum_arr = np.cumsum(channel, axis=0)
        cumsum_arr = (cumsum_arr / cumsums_max) * 255.
        return cumsum_arr

    def depth(self, record):
        mode, image_id = record.split('-', 1)
        return self.depths[image_id]

dataset = SaltDataset(DIRECTORY)
#dataset.check_integrity() NOT WORKING
dataset.preprocessors = { 'image': img_preprocess, 'mask': mask_preprocess }
dataset.postprocessors = { 'image': img_postprocess, 'mask': mask_postprocess }

def train_step(model, dataset, step_index=0, model_name='', **options):
    save_filepath=SAVE_PATH + (model_name % step_index)

    log_start = ("##  Train step %i: %s" % (step_index, save_filepath))
    split = options.pop('split', None)
    if split:
        options['split'] = type(split)
    printable_options = "\n##    ".join([("%s: %s" % (str(k).ljust(12), v)) for k, v in options.items()])
    log_start += ("\n##  Options:\n##    %s" % printable_options)
    print(log_start)
    if split:
        options['split'] = split

    checkpointer = Best(filepath=save_filepath)
    log_filename = save_filepath.replace('_{epoch:02d}_{val_loss:.4f}.pt', '.log.csv')
    with open(log_filename, 'a') as f:
        f.write(log_start + "\n")
    logger = CSVLogger(filename=log_filename, append=True)
    sched = schedulers.ReduceLROnPlateau(patience=2, verbose=True)
    options['callbacks'] = [checkpointer, logger, sched]

    [model.freeze(layer_name) for layer_name in options.pop('freeze', [])]
    [model.unfreeze(layer_name) for layer_name in options.pop('unfreeze', [])]

    train(model, dataset, **options)

from sklearn.model_selection import train_test_split
def split_data(dataset):
    split = {}
    if dataset.__dict__.get('stratify'):
        split['stratify'] = [getattr(dataset, dataset.stratify)(record) for record in dataset.train]

    return tuple(train_test_split(dataset.train, **split))

def train_sequence(dataset, model=None, steps=[], **options):
    if not model:
        model = MODELS[MODEL_NAME](3, load_weights=LOAD_WEIGHTS, weights_path=WEIGHTS_PATH)
        timed_model_name = id_generator(MODEL_NAME)
        print(timed_model_name)

    options['split'] = split_data(dataset)
    for i, step in enumerate(steps):
        for k, v in step.items():
            options[k] = v
        opts = copy.deepcopy(options)
        train_step(model, dataset, step_index=i, model_name=timed_model_name, **opts)


# TRAINING

print("Simple default params training")
steps = [
    {}
]
custom_options = {
    'epochs': 5,
    'optimizer': (OPT, { 'lr': 1e-4 })
}
train_sequence(dataset, steps=steps, **merge_options(custom_options))
print("Looong training")
steps = [
    {}
]
custom_options = {
    'epochs': 100,
    'optimizer': (OPT, { 'lr': 1e-5 })
}
train_sequence(dataset, steps=steps, **merge_options(custom_options))
print("Looong training with augment")
steps = [
    {}
]
custom_options = {
    'augment': { ('image', 'mask'): [
        {'type': 'PadIfNeeded', 'min_height': 128, 'min_width': 202},
        {'type': 'RandomCrop', 'height': 128, 'width': 128},
        {'type': 'HorizontalFlip'},
    ] },
    'epochs': 100,
    'optimizer': (OPT, { 'lr': 1e-5 })
}
train_sequence(dataset, steps=steps, **merge_options(custom_options))
print("Looong training with augment without weights")
LOAD_WEIGHTS = False
steps = [
    {}
]
custom_options = {
    'augment': { ('image', 'mask'): [
        {'type': 'PadIfNeeded', 'min_height': 128, 'min_width': 202},
        {'type': 'RandomCrop', 'height': 128, 'width': 128},
        {'type': 'HorizontalFlip'},
    ] },
    'epochs': 100,
    'optimizer': (OPT, { 'lr': 1e-5 })
}
train_sequence(dataset, steps=steps, **merge_options(custom_options))
