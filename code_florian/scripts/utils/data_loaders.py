import os
import cv2
import sys
import glob
import math
import time
import random
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any

from scipy import ndimage
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize
import skimage.filters as fltrs

import torch
import torchvision as vsn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils.transforms import augment_img, reflect_pad, mt_noise

#cv2.setNumThreads(0)

def add_depth_channels(image_tensor, mosaic_mode):
    _, h, w = image_tensor.size()

    if mosaic_mode != 1:
        for row, const in enumerate(np.linspace(0, 1, h)):
            image_tensor[1, row, :] = const

    if mosaic_mode != 2:
        for row, const in enumerate(np.linspace(0, 1, h)):
            image_tensor[2, row, :] = image_tensor[0, row, :] * const

    return image_tensor

def parse_mosaic():
    ''' reads adjacency information. Returns dicts: [left, top, right, bottom] '''
    left, top, right, bottom = dict(), dict(), dict(), dict()
    print("loading mosaic")

    for slice in tqdm(glob.glob("../data/mos_numpy/*.csv")):
        mosaic = pd.read_csv(slice, header=None).values

        for x, y in np.ndindex(mosaic.shape):
            img = mosaic[x, y]

            def non_empty(val: Any) -> bool:
                return type(val) == str

            if x > 0 and non_empty(mosaic[x - 1, y]):
                left[img] = mosaic[x - 1, y]

            if y > 0 and non_empty(mosaic[x, y - 1]):
                top[img] = mosaic[x, y - 1]

            if x < mosaic.shape[0] - 1 and non_empty(mosaic[x + 1, y]):
                right[img] = mosaic[x + 1, y]

            if y < mosaic.shape[1] - 1 and non_empty(mosaic[x, y + 1]):
                bottom[img] = mosaic[x, y + 1]

    return [left, top, right, bottom]

def add_neighbours(mosaic_mode, mosaic, img, id):
    """ Takes a batch (N, 101, 101, 3) and list of id's. Adds mosaic data."""
    if mosaic_mode not in [1, 2]:
        return img

    img[:, :, mosaic_mode] = 0.5
    name = id[:-4]

    # mask_names = ["left", "top", "right", "bottom"]
    mask_offsets = [(0, 0), (50, 0), (0, 50), (50, 50)]

    for neighbour in range(4):
        if name in mosaic[neighbour]:
            path = "../data/train/masks/" + mosaic[neighbour][name] + ".png"

            if os.path.exists(path):
                neigh = img_as_float(imread(path))
                # print("found neighbour", mask_names[neighbour], neigh.shape)
                neigh = resize(img, (50, 50))[:, :, 0]

                ofs = mask_offsets[neighbour]
                print(neigh.shape)
                img[ofs[0] : ofs[0]+50, ofs[1] : ofs[1]+50, mosaic_mode] = neigh

    return img

def add_edges_channel(image_tensor, edges_channel):
    image_tensor[3] = edges_channel
    return image_tensor

def edges(mask, threshold=0.5):
    #import pdb
    #pdb.set_trace()
    mask = mask > threshold
    struct = ndimage.generate_binary_structure(3, 2)
    erode = ndimage.binary_erosion(mask, struct)
    edges = mask ^ erode
    return edges

class MaskDataset(data.Dataset):
    '''Generic dataloader for a pascal VOC format folder'''
    def __init__(self, mosaic, mosaic_mode, imsize=128, img_ids=None, img_paths=None,
                 mask_paths=None, valid=False, small_msk_ids=None):
        self.valid = valid
        self.imsize = imsize
        self.img_ids = img_ids
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.small_msk_ids = small_msk_ids
        self.mosaic = mosaic
        self.mosaic_mode = mosaic_mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        # super sample small masks
        #if random.random() > 0.25 or self.valid:
        img = img_as_float(imread(self.img_paths + self.img_ids[index]))[:,:,:3]
        img = add_neighbours(self.mosaic_mode, self.mosaic, img, self.img_ids[index])
        #img[:,:,2] = 1. - fltrs.laplace(img[:,:,1])

        msk = imread(self.mask_paths + self.img_ids[index]).astype(np.bool)
        msk = np.expand_dims(msk, axis=-1)
        edgs = edges(msk)
        #else:
        #    small_idx = random.randint(0, len(self.small_msk_ids))
        #    img = img_as_float(imread('../data/train/small_masks/images/' + self.small_msk_ids[small_idx]))[:,:,:3]
        #    msk = imread('../data/train/small_masks/masks/' + self.small_msk_ids[small_idx]).astype(np.bool)
        #    msk = np.expand_dims(msk, axis=-1)

        if not self.valid:
            img_np, msk_np, edg_np  = augment_img([img, msk, edgs], imsize=self.imsize)
            img_lr = np.fliplr(img_np)
        else:
            #img_np = resize(np.asarray(img), (self.imsize, self.imsize),
            #            preserve_range=True, mode='reflect')
            #msk_np = resize(msk, (self.imsize, self.imsize),
            #                preserve_range=True, mode='reflect')
            img_np = reflect_pad(img, int((self.imsize - img.shape[0]) / 2))
            img_lr = np.fliplr(img_np)
            msk_np = reflect_pad(msk, int((self.imsize - msk.shape[0]) / 2))
            edg_np = reflect_pad(edgs, int((self.imsize - edgs.shape[0]) / 2))
            img_np = img_np.transpose((2,0,1)).astype(np.float32)
            img_lr = img_lr.transpose((2,0,1)).astype(np.float32)
            msk_np = msk_np.transpose((2,0,1)).astype(np.float32)
            edg_np = edg_np.transpose((2,0,1)).astype(np.float32)

        # get image ready for torch
        edg_tch = torch.from_numpy(edg_np.astype(np.float32))
        img_tch = self.normalize(torch.from_numpy(img_np.astype(np.float32)))
        msk_tch = torch.from_numpy(msk_np.astype(np.float32))
        img_tch = add_depth_channels(img_tch, self.mosaic_mode)

        img_lr_tch = self.normalize(torch.from_numpy(img_lr.astype(np.float32)))
        img_lr_tch = add_depth_channels(img_lr_tch, self.mosaic_mode)

        out_dict = {'img': img_tch,
                    'msk': msk_tch,
                    'edges': edg_tch,
                    'has_msk': msk_tch.sum() > 0,

                    'img_lr': img_lr_tch,
                    'blank': torch.tensor(os.stat('../data/train/images/' + self.img_ids[index]).st_size != 107),

                    'id': self.img_ids[index].replace('.png', '')}

        return out_dict

    def __len__(self):
        if self.valid:
            return int(len(self.img_ids) * 0.2)
        else:
            return int(len(self.img_ids) * 0.8)


class MaskDataset_MT(data.Dataset):
    '''Generic dataloader for a pascal VOC format folder'''
    def __init__(self, imsize=128, labeled_ids=None, labeled_img_paths=None,
                 unlabeled_index=0, unlabeled_ids=None, unlabeled_img_paths=None,
                 unlabeled_ratio=0.5, mask_paths=None):
        self.imsize = imsize
        self.labeled_ids = labeled_ids
        self.labeled_img_paths = labeled_img_paths
        self.unlabeled_idx = unlabeled_index
        self.unlabeled_ids= unlabeled_ids
        self.unlabeled_ratio = unlabeled_ratio
        self.unlabeled_img_paths = unlabeled_img_paths
        self.mask_paths = mask_paths
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        random.seed()

        # choose whether to pull labeled or unlabaled
        labeled = 1 if random.random() > self.unlabeled_ratio else 0 # print(imgs.size())
        if labeled == 1:
            img = img_as_float(imread(self.labeled_img_paths +
                                      self.labeled_ids[index]))[:,:,:3]
            msk = imread(self.mask_paths + self.labeled_ids[index]).astype(np.bool)
            msk = np.expand_dims(msk, axis=-1)
        else:
            img = img_as_float(imread(self.unlabeled_img_paths +
                                      self.unlabeled_ids[self.unlabeled_idx]))[:,:,:3]
            msk = np.ones((101, 101, 1)) * -1.
            self.unlabeled_idx += 1
            # if we go through all the labeled images, shuffle them and start counter over
            if self.unlabeled_idx >= len(self.unlabeled_ids):
                # self.unlabeled_ids = shuffle(self.unlabeled_ids) # unknown symbol
                self.unlabeled_idx = 0

        # the geometric augmentions have to be the same
        img, msk = augment_img([img, msk], imsize=self.imsize, mt=True)
        # brightness, gamma, and gaussian noise can be different
        img_a = mt_noise(img)
        img_b = mt_noise(img)

        msk_tch = torch.from_numpy(msk.astype(np.float32))

        out_dict = {'img_a': self.normalize(torch.from_numpy(img_a.astype(np.float32))),
                    'img_b': self.normalize(torch.from_numpy(img_b.astype(np.float32))),
                    'msk': msk_tch,
                    'has_msk': msk_tch.sum() > 0,
                    'is_labeled': torch.tensor(labeled).long()}

        return out_dict

    def __len__(self):
        return int(len(self.labeled_ids) * 0.8)

class MaskTestDataset(data.Dataset):
    '''Dataset for loading the test Images'''
    def __init__(self, mosaic, mosaic_mode, imsize=128, img_ids=None, img_paths=None):
        self.imsize = imsize
        self.img_ids = img_ids
        self.img_paths = img_paths
        self.mosaic = mosaic
        self.mosaic_mode = mosaic_mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        img = img_as_float(imread('../data/test/images/' + self.img_ids[index]))[:,:,:3]
        img = add_neighbours(self.mosaic_mode, self.mosaic, img, self.img_ids[index])

        # scale up image to 202 or keep at 101, reflect pad to get network sizes
        if self.imsize == 256:
            img = resize(img, (202, 202), preserve_range=True, mode='reflect')
            img = reflect_pad(img, 27)
        else:
            img = reflect_pad(img, 13)
        img_lr = np.fliplr(img)

        #print(img.shape, img_lr.shape)

        img = img.transpose((2,0,1)).astype(np.float32)
        img_lr = img_lr.transpose((2,0,1)).astype(np.float32)

        img_tch = self.normalize(torch.from_numpy(img))
        img_lr_tch = self.normalize(torch.from_numpy(img_lr))

        img_tch = add_depth_channels(img_tch, self.mosaic_mode)
        img_lr_tch = add_depth_channels(img_lr_tch, self.mosaic_mode)

        out_dict = {'img': img_tch,
                    'img_lr': img_lr_tch,
                    'id': self.img_ids[index].replace('.png', ''),
                    'blank': torch.tensor(os.stat('../data/test/images/' + self.img_ids[index]).st_size != 107)}

        return out_dict

    def __len__(self):
        return len(self.img_ids)

def get_data_loaders(imsize=128, batch_size=16, num_folds=5, fold=0, mosaic_mode=0, num_workers=8):
    '''sets up the torch data loaders for training'''
    mosaic = parse_mosaic()

    with open("../data/fixed_files.pkl", "rb") as f:
        img_ids = pickle.load(f)

    img_idx = list(range(len(img_ids)))

    with open("../data/fixed_folds.pkl", "rb") as f:
        splits = pickle.load(f)

    train_idx, valid_idx = splits[fold]

    small_msk_ids = list(set([os.path.basename(x) for x in glob.glob('../data/train/small_masks/images/*.png')]) &
                         set([img_ids[idx] for idx in train_idx]))

    print('Supersampling {} small masks'.format(len(small_msk_ids)))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # set up the datasets
    train_dataset = MaskDataset(mosaic, mosaic_mode,
                                imsize=imsize, img_ids=img_ids,
                                img_paths='../data/train/images/',
                                mask_paths='../data/train/masks/',
                                small_msk_ids=small_msk_ids)
    valid_dataset = MaskDataset(mosaic, mosaic_mode,
                                imsize=imsize, img_ids=img_ids,
                                img_paths='../data/train/images/',
                                mask_paths='../data/train/masks/',
                                valid=True)

    # set up the data loaders
    train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       sampler=train_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)

    valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       sampler=valid_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)

    return train_loader, valid_loader

def get_data_mt_loaders(imsize=128, batch_size=16, num_folds=5, fold=0, unlabeled_ratio=0.5):
    '''sets up the torch data loaders for training'''
    with open("../data/fixed_files.pkl", "rb") as f:
        img_ids = pickle.load(f)

    img_idx = list(range(len(img_ids)))

    with open("../data/fixed_folds.pkl", "rb") as f:
        splits = pickle.load(f)

    train_idx, valid_idx = splits[fold]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    unlabeled_ids = [os.path.basename(x) for x in glob.glob('../data/test/images/*.png')]

    # set up the datasets
    train_dataset = MaskDataset_MT(imsize=imsize, labeled_ids=img_ids,
                                   unlabeled_ids=unlabeled_ids,
                                   unlabeled_ratio=unlabeled_ratio,
                                   labeled_img_paths='../data/train/images/',
                                   unlabeled_img_paths='../data/test/images/',
                                   mask_paths='../data/train/masks/')

    valid_dataset = MaskDataset(imsize=imsize, img_ids=img_ids,
                                  img_paths='../data/train/images/',
                                  mask_paths='../data/train/masks/',
                                  valid=True)

    # set up the data loaders
    train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       sampler=train_sampler,
                                       num_workers=4,
                                       pin_memory=True,
                                       drop_last=True)

    valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       sampler=valid_sampler,
                                       num_workers=4,
                                       pin_memory=True)

    return train_loader, valid_loader


def get_test_loader(imsize=128, batch_size=16, mosaic_mode=0):
    '''sets up the torch data loaders for training'''
    mosaic = parse_mosaic()

    img_ids = [os.path.basename(x) for x in glob.glob('../data/test/images/*.png')]
    print('Found {} test images'.format(len(img_ids)))

    # set up the datasets
    test_dataset = MaskTestDataset(mosaic, mosaic_mode,
                                   imsize=imsize, img_ids=img_ids,
                                   img_paths='../data/test/images/')

    # set up the data loaders
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=False)

    return test_loader

if __name__ == '__main__':
    train_loader, valid_loader = get_data_loaders(imsize=128, batch_size=32)

    for i, data in enumerate(train_loader):
        if i == 1:
            break
        img = data['img']
        msk = data['msk']

        img_grid = vsn.utils.make_grid(img, normalize=True)
        msk_grid = vsn.utils.make_grid(msk)

        vsn.utils.save_image(img_grid, '../imgs/train_imgs.png')
        vsn.utils.save_image(msk_grid, '../imgs/train_msks.png')
