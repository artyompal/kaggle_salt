#!/usr/bin/python3.6

import os
from glob import glob
import pandas as pd


if __name__ == "__main__":
    train = glob("../data/train/images/*.png")
    test = glob("../data/test/images/*.png")
    depth = pd.read_csv("../data/depths.csv", index_col="id")
    depths = set(depth.index)
    print("train:", len(train), "test:", len(test), "depths:", len(depths))

    for image in train + test:
        name = os.path.basename(image)[:-4]

        if not name in depths:
            print(name)
