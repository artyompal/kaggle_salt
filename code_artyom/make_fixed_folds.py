#!/usr/bin/python3.6

import os, pickle
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from skimage.transform import resize
from keras.preprocessing.image import array_to_img, img_to_array, load_img


NUM_FOLDS       = 5

img_size_ori    = 101
img_size_target = 101


def upsample(img): # type: ignore
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

def cov_to_class(val): # type: ignore
    for i in range(0, 11):
        if val * 10 <= i :
            return i

if __name__ == "__main__":
    print(f"using {NUM_FOLDS} folds")

    # Loading of training/testing ids and depths
    train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("../data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)

    train_df["images"] = [np.array(load_img("../data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
    train_df["masks"] = [np.array(load_img("../data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

    images = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    masks = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

    folds = StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=666)
    fixed_folds = []

    for fold, indices in enumerate(folds.split(images, train_df.coverage_class)):
        train_idx, valid_idx = indices
        fixed_folds.append((train_idx, valid_idx))

    print(fixed_folds)
    path = "../data/fixed_folds.pkl"

    with open(path, "wb") as f:
        pickle.dump(fixed_folds, f)

    print(f"pickle with all folds is written into {path}.")
    print("The format is: list of 5 tuples (train_idx, val_idx)")
