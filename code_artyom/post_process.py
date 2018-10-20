#!/usr/bin/python3.6

import os, pickle, random, subprocess, sys
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from skimage.transform import resize

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from keras.preprocessing.image import load_img


img_size_ori    = 101
img_size_target = 101


# def enable_logging() -> None:
#     """ Sets up logging to a file. """
#     module_name = os.path.splitext(os.path.basename(__file__))[0]
#     log_file = '../output/' + module_name + ".log"
#
#     tee = subprocess.Popen(["tee", "-a", log_file], stdin=subprocess.PIPE)
#     os.dup2(tee.stdin.fileno(), sys.stdout.fileno())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} dest.csv source.csv")
        sys.exit()

    # enable_logging()
    dest_submission = sys.argv[1]
    source_submission = sys.argv[2]

    sub = pd.read_csv(source_submission, index_col="id")

    train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("../data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    x_test = [np.array(load_img("../data/test/images/{}.png".format(idx), grayscale=True))
              / 255 for idx in tqdm(test_df.index)]
    is_empty = [np.sum(img) == 0 for img in x_test]
    sub["rle_mask"][is_empty] = ""

    sub.to_csv(dest_submission)
