#!/usr/bin/python3.6

import glob, math, sys
import numpy as np, pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import load_img

def get_class(img, th=10): # type: ignore
    img_sum = np.array([i.sum() for i in img])
    return np.array(img_sum > th).astype(int)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"usage: {sys.argv[0]} dest.csv prediction.csv train_classes.csv test_classes.csv")
        sys.exit()

    train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
    train_classes = pd.read_csv(sys.argv[3], index_col="id")
    assert(train_classes.shape[0] == 4000)
    test_classes = pd.read_csv(sys.argv[4], index_col='id')
    assert(test_classes.shape[0] == 18000)
    # print(test_classes)

    print("reading masks")
    masks = [np.array(load_img(f"../data/train/masks/{idx}.png", color_mode="grayscale"))
             for idx in train_df.index]
    ground_truth = {idx: cls for idx, cls in zip(train_df.index, get_class(masks))} # type: ignore
    print("number of non-empty masks", sum(ground_truth.values()))

    # Searching for the best threshold
    N = 100
    candidates = np.linspace(np.amin(train_classes['class']), np.amax(train_classes['class']), N)
    accuracies = []

    for th in tqdm(candidates):
        images = train_classes.index.values
        predictions = (train_classes['class'].values > th).astype(int)

        # print('images', images)
        # print('predictions', predictions)

        # accuracies.append(np.sum(predictions == ground_truth) / predictions.shape[0])
        accuracies.append(sum([ground_truth[images[i]] == pred
                               for i, pred in enumerate(predictions)]))

    accuracies = np.array(accuracies) / train_df.shape[0]
    best_idx = np.argmax(accuracies)
    accuracy = accuracies[best_idx]
    threshold = candidates[best_idx]

    print("accuracies", accuracies)
    print("accuracy", accuracy)
    print("threshold", threshold)


    # The classifier returns 0 for empty masks, 1 for non-empty ones.
    # EMPTY_MASKS_FRACTION = 38
    # threshold = np.percentile(test_classes['class'].values, EMPTY_MASKS_FRACTION)
    # print("threshold", threshold)
    print("below", np.sum(test_classes['class'].values < threshold))
    print("above", np.sum(test_classes['class'].values >= threshold))
    print("fraction", np.sum(test_classes['class'].values < threshold) / test_classes.shape[0])

    probabilities = test_classes.to_dict()['class']

    sub = pd.read_csv(sys.argv[2])
    new_values = sub.rle_mask.values

    for index, row in sub.iterrows():
        id, rle_mask = row['id'], row['rle_mask']

        if probabilities[id] < threshold:
            new_values[index] = "1 1"
        else:
            new_values[index] = ""

    sub['rle_mask'] = new_values
    # print(sub)
    sub.to_csv(sys.argv[1], index=False)
