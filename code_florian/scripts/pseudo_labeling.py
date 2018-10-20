#!/usr/bin/python3.6

import argparse, pickle
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from typing import Any
from glob import glob
from tqdm import tqdm
from skimage.io import imread


def rle_encode(im: Any) -> str:
    '''
    im: numpy array, 1-mask, 0-background
    Returns run length as string
    '''
    assert(im.shape[0] == 101 and im.shape[1] == 101)
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pseudo-labeling')
    parser.add_argument('--cutoff', default=0.3, type=float,
                        help='confidence cut-off value (0 to 0.5)')
    args = parser.parse_args()

    filenames = list(glob("../best_models/loc*_test_*.pkl"))
    print('found %d predicts to merge' % len(filenames))
    for file in filenames:
        print(file)

    predicts = np.zeros((18000, 101, 101))

    for path in tqdm(filenames):
        with open(path, "rb") as f:
            ids, preds = pickle.load(f)

        ids, preds = list(ids), list(preds)
        df = pd.DataFrame.from_dict({'id':ids, 'pred':preds}) #, orient='index')
        df = df.sort_values("id")

        predicts += np.array(df.pred.tolist())
        ids = df.id.values

    predicts /= len(filenames)
    print("predicts", predicts.shape)
    std = np.std(predicts, axis=(1, 2))
    print("std", std.shape)
    print("ids", ids.shape)

    if False:
        plt.hist(std, bins='auto')
        plt.title("Confidence distribution")
        plt.show()

    predicts = predicts[std > args.cutoff]
    ids = ids[std > args.cutoff]
    print("predicts after cutoff", predicts.shape)

    print("predicts after cast", predicts.shape)
    print("encoding predictions")
    rles = np.array([rle_encode(np.where(p > 0.5, 1, 0)) for p in tqdm(predicts)])
    print("rles", rles.shape)

    pseudo_labels = pd.DataFrame({'id': ids, 'rle_mask': rles})
    pseudo_labels.to_csv("../data/pseudo_labels.csv", index=False)
    # print(pseudo_labels)

    if False:
        for _ in range(20):
            idx = np.random.randint(predicts.shape[0])

            plt.subplot(2, 2, 1)
            plt.imshow(predicts[idx])
            plt.subplot(2, 2, 2)
            plt.imshow(imread(f"../data/test/images/{ids[idx]}.png"))
            plt.show()
