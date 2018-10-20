#!/usr/bin/python3.6

import glob, math, sys
import numpy as np, pandas as pd
from tqdm import tqdm

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(101, 101)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    if str(mask_rle) != str(np.nan):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def rle_encode(im):
    '''
    im: numpy array, 1-mask, 0-background
    Returns run length as string
    '''
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"usage: {sys.argv[0]} dest.csv 1.csv 2.csv 3.csv...")
        sys.exit()

    dest_csv = sys.argv[1]
    submissions = sys.argv[2:]
    print('found {} submissions to merge'.format(len(submissions)))

    folds = []
    for subm in submissions:
        df = pd.read_csv(subm)
        df = df.sort_values("id")
        folds.append(df)

    print(folds[0].head())

    ids = []
    rles = []

    for i in tqdm(range(len(folds[0]))):
        mask = np.zeros((101, 101))

        for j in range(len(folds)):
            mask += rle_decode(folds[j].iloc[i]['rle_mask'])

        # majority vote for ensembled mask
        mask = np.where(mask >= np.round(len(folds) * 0.5), 1, 0)

        ids.append(folds[0].iloc[i]['id'])
        rles.append(rle_encode(mask.astype(np.int32)))

    ensmb_preds = pd.DataFrame({'id': ids, 'rle_mask': rles})
    ensmb_preds.head()
    ensmb_preds.to_csv(dest_csv, index=False)
