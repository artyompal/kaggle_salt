#!/usr/bin/python3.6

import multiprocessing, os, pickle, random, subprocess, sys
from functools import partial
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from skimage.transform import resize

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from keras.preprocessing.image import load_img


NUM_FOLDS       = 5
PREDICT_ONLY    = True

img_size_ori    = 101
img_size_target = 101


def enable_logging() -> None:
    """ Sets up logging to a file. """
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    log_file = '../output/' + module_name + ".log"

    tee = subprocess.Popen(["tee", "-a", log_file], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    # os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def make_output_path(filename: str) -> str:
    """ Returns a correct file path to save to. """
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    name_ext = os.path.splitext(filename)
    return '../output/' + name_ext[0] + '_' + module_name + name_ext[1]

def upsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

def downsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i


#Score the model and do a threshold optimization by the best IoU.

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in


    true_objects = 2
    pred_objects = 2

    #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
    intersection = temp1[0]
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    assert(y_true_in.shape[0] == y_pred_in.shape[0])
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

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

def calc_iou_for_threshold(ground_truth, pred, threshold):
    return iou_metric_batch(ground_truth, pred > threshold)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} submission.csv min_score models...")
        sys.exit()

    enable_logging()
    print(f"using {NUM_FOLDS} folds")
    submission_file = sys.argv[1]
    min_score = float(sys.argv[2])
    models = sys.argv[3:]

    # Loading of training/testing ids and depths
    train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("../data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    train_df["images"] = [np.array(load_img("../data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
    train_df["masks"] = [np.array(load_img("../data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)


    images = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    masks = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

    folds = StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=666)
    x_test = np.array([(np.array(load_img("../data/test/images/{}.png".format(idx), grayscale = True))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)

    print("train", images.shape)
    print("coverage_class", train_df.coverage_class.shape)

    final_pred = np.zeros((test_df.shape[0], img_size_target, img_size_target))
    final_count = 0
    pool = multiprocessing.Pool()

    for model_name in models:
        print("using predicts for", model_name)

        for fold, indices in enumerate(folds.split(images, train_df.coverage_class)):
            print("==================== fold %d" % fold)
            train_idx, valid_idx = indices

            x_train, y_train = images[train_idx], masks[train_idx]
            x_valid, y_valid = images[valid_idx], masks[valid_idx]

            with open("../output/predicts/fold%d_test_%s.pkl" % (fold, model_name), "rb") as f:
                pred_test = pickle.load(f)

            with open("../output/predicts/fold%d_train_%s.pkl" % (fold, model_name), "rb") as f:
                pred_valid = pickle.load(f)

            # Scoring for last model, choose threshold by validation data
            thresholds_ori = np.linspace(0.3, 0.7, 31)

            # Reverse sigmoid function: use code below because the sigmoid activation was removed
            thresholds = np.log(thresholds_ori/(1-thresholds_ori))

            print("y_valid", y_valid.shape)
            print("pred_valid", pred_valid.shape)
            # ious = np.array([iou_metric_batch(y_valid, pred_valid > threshold) for threshold in tqdm(thresholds)])

            calc = partial(calc_iou_for_threshold, y_valid, pred_valid)
            ious = np.array(pool.map(calc, thresholds))

            threshold_best_index = np.argmax(ious)
            iou_best = ious[threshold_best_index]
            threshold_best = thresholds[threshold_best_index]
            print("validation:", iou_best, "threshold", threshold_best)

            if iou_best < min_score:
                print("skipping")
                continue

            pred_test = np.round(pred_test > threshold_best)
            final_pred += pred_test
            final_count += 1

    pool.close()
    pool.terminate()

    final_pred /= final_count
    final_threshold = 0.5
    pred_dict = {idx: rle_encode(np.round(downsample(final_pred[i]) > final_threshold))
                 for i, idx in enumerate(tqdm(test_df.index.values))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(submission_file)
