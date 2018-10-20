#!/usr/bin/python3.6

# ### U-net with simple Resnet Blocks v2, can get 0.80+
# * Original version :
#   https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks
#
#
# #### update log
# 1.   Cancel last dropout (seems better)
# 2.  modify convolution_block, to be more consistant with the standard resent model.
#       * https://arxiv.org/abs/1603.05027
# 3. Use faster  IOU metric score code,
#       * https://www.kaggle.com/donchuk/fast-implementation-of-scoring-metric
# 4. Use  binary_crossentropy loss and then Lovász-hinge loss (very slow!)
#      * Lovász-hinge loss: https://github.com/bermanmaxim/LovaszSoftmax
#
# Limit the max epochs number to make the kernel finish in the limit of 6 hours, better score can be achived at more epochs

import os, pickle, random, sys, subprocess, tracemalloc
from typing import *

import numpy as np, pandas as pd, scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold

from tqdm import tqdm
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing

from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers

import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img


NpArray = Any

ENABLE_KFOLD = False
EPOCHS      = 50
BATCH_SIZE  = 32
VERBOSE     = 2
NUM_FOLDS   = 5 if ENABLE_KFOLD else 1

img_size_ori = 101
img_size_target = 224


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

def upsample(img: NpArray) -> NpArray:
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant',
                  preserve_range=True, anti_aliasing=True)

def downsample(img: NpArray) -> NpArray:
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant',
                  preserve_range=True, anti_aliasing=True)

def batch_downsample(batch: NpArray) -> NpArray:
    if img_size_ori == img_size_target:
        return batch

    print("downsampling")
    res = np.array([resize(img, (img_size_ori, img_size_ori), mode='constant',
                          preserve_range=True, anti_aliasing=True)
                   for img in tqdm(batch)])
    print("shape after downsample", res.shape)
    return res

def cov_to_class(val: float) -> int:
    return int(val / 0.1)

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0], tf.float64)

# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss

def flatten_binary_scores(scores, labels, ignore=None):
    """ Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore' """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels

    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image=True, ignore=None)
    return loss

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
        true_positives = np.sum(matches, axis=1) == 1   # correct objects
        false_positives = np.sum(matches, axis=0) == 0  # missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # extra objects
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

def iou_metric_batch(y_true: NpArray, y_pred: NpArray) -> float:
    assert(y_true.shape[0] == y_pred.shape[0])
    batch_size = y_true.shape[0]
    metric = []

    for batch in range(batch_size):
        value = iou_metric(y_true[batch], y_pred[batch])
        metric.append(value)

    return np.mean(metric)

def rle_encode(im) -> str:
    """ Converts the decoded image into RLE mask.
    im: numpy array, 1 - mask, 0 - background """
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def train(x_train: NpArray, x_valid: NpArray, y_train: NpArray, y_valid: NpArray,
          fold: int = -1) -> None:
    preprocessing_fn = get_preprocessing('resnet34')
    x_train = preprocessing_fn(x_train)
    x_valid = preprocessing_fn(x_valid)

    model = Unet(backbone_name='resnet34', encoder_weights='imagenet')
    model.compile('Adam', 'binary_crossentropy', metrics=[my_iou_metric])
    model.summary()

    model_name = make_output_path("models/fold%d.hdf5" % fold)
    model_checkpoint = ModelCheckpoint(model_name, monitor='val_my_iou_metric',
                                       mode='max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max',
                                  factor=0.5, patience=5, min_lr=3e-6, verbose=1)

    model.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=EPOCHS,
              batch_size=BATCH_SIZE, callbacks=[model_checkpoint, reduce_lr],
              verbose=VERBOSE)

def predict_result(model: Any, x_test: NpArray, img_size_target: NpArray) -> NpArray:
    """ Predicts for both orginal and reflected dataset. """
    x_test_reflect = np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test, verbose=VERBOSE)
    preds_test = preds_test.reshape(-1, img_size_target, img_size_target)
    preds_test2_reflect = model.predict(x_test_reflect, verbose=VERBOSE)
    preds_test2_reflect = preds_test2_reflect.reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_reflect] )
    return preds_test/2

def predict(x_valid: NpArray, x_test: NpArray, fold: int = -1) -> Tuple[NpArray, NpArray]:
    model_name = make_output_path("models/fold%d.hdf5" % fold)
    model = load_model(model_name,
                       custom_objects={'my_iou_metric': my_iou_metric,
                                       'my_iou_metric_2': my_iou_metric_2,
                                       'lovasz_loss': lovasz_loss})

    print("predicting...")
    preds_valid = predict_result(model, x_valid, img_size_target)
    preds_test = predict_result(model, x_test, img_size_target)
    return preds_valid, preds_test

def generate_submission(preds_valid: NpArray, ground_truth_valid: NpArray,
                        preds_test: NpArray) -> None:
    # Score the model and do a threshold optimization by the best IoU.
    print("preds_valid", preds_valid.shape, "preds_test", preds_test.shape)

    # Scoring for last model, choose threshold by validation data
    thresholds_ori = np.linspace(0.3, 0.7, 31)

    # Reverse sigmoid function: use code below because the sigmoid activation was removed
    thresholds = np.log(thresholds_ori / (1 - thresholds_ori))

    print("searching threshold")
    ious = np.array([iou_metric_batch(ground_truth_valid, preds_valid > threshold)
                     for threshold in tqdm(thresholds)])
    print("ious", ious)

    # instead of using default 0 as threshold, use validation data to find the best threshold.
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    print("validation metric:", iou_best)
    print("best threshold:", threshold_best)

    print("encoding prediction")
    pred_dict = {idx: rle_encode(np.round(preds_test[i]) > threshold_best)
                 for i, idx in enumerate(tqdm(test_df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(make_output_path("sub.csv"))

def load_train_data() -> Tuple[NpArray, NpArray, NpArray, NpArray, Any]:
    train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("../data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    print("train_df", train_df.shape, "test_df", test_df.shape)

    print("reading train images")
    train_df["images"] = [np.array(load_img("../data/train/images/%s.png" % idx))
                          / 255 for idx in tqdm(train_df.index)]
    images = np.array(train_df.images.map(upsample).tolist())
    print("images", images.shape)

    print("reading train masks")
    train_df["masks"] = [np.array(load_img("../data/train/masks/%s.png" % idx,
                                           color_mode="grayscale"))
                         / 255 for idx in tqdm(train_df.index)]
    masks = np.array(train_df.masks.map(upsample).tolist())
    masks_orig = np.array(train_df.masks.tolist())
    masks = np.expand_dims(masks, axis=-1)
    masks_orig = np.expand_dims(masks_orig, axis=-1)
    print("masks", masks.shape)
    print("masks_orig", masks_orig.shape)

    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    labels_for_strat = train_df.coverage_class
    print("labels_for_strat", labels_for_strat.shape)

    return images, masks, masks_orig, labels_for_strat, test_df

def load_test_data(test_df: Any) -> NpArray:
    print("reading test set")
    x_test = np.array([upsample(np.asarray(load_img("../data/test/images/%s.png" % idx ))) / 255
                       for idx in tqdm(test_df.index)])
    print("x_test", x_test.shape)
    return x_test

if __name__ == "__main__":
    enable_logging()
    if ENABLE_KFOLD:
        print(f"training with {NUM_FOLDS} folds")
    else:
        print("training without folds")

    images, masks, masks_lowres, labels_for_strat, test_df = load_train_data()

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")
    # print("memory stats:")
    # for stat in top_stats[:10]:
    #     print(stat)

    if not ENABLE_KFOLD:
        x_train, x_valid, y_train, y_valid, _, y_valid_lowres = \
            train_test_split(images, masks, masks_lowres, stratify=labels_for_strat,
                             shuffle=True, random_state=666, test_size=0.2)

        train(x_train, x_valid, y_train, y_valid)

        x_test = load_test_data(test_df)
        preds_valid, preds_test = predict(x_valid, x_test)
        ground_truth_valid = y_valid_lowres
    else:
        preds_train = np.zeros((images.shape[0], img_size_target, img_size_target))
        preds_test = np.zeros((NUM_FOLDS, x_test.shape[0], img_size_target, img_size_target))
        print("preds_train", preds_train.shape)
        print("preds_test", preds_test.shape)

        x_test = load_test_data(test_df)
        folds = StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=666)

        for fold, indices in enumerate(folds.split(images, labels_for_strat)):
            print("==================== fold %d" % fold)

            train_idx, valid_idx = indices
            x_train, y_train = images[train_idx], masks[train_idx]
            x_valid, y_valid = images[valid_idx], masks[valid_idx]

            # # data augmentation
            # x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
            # y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

            train(x_train, x_valid, y_train, y_valid, fold)

            p_val, p_test = predict(x_valid, x_test, fold)
            preds_train[valid_idx], preds_test[fold] = p_val, p_test

            with open(make_output_path("predicts/fold%d_test.pkl" % fold), "wb") as f:
                pickle.dump(p_test, f)

        preds_valid = preds_train
        ground_truth_valid = masks
        preds_test = np.mean(preds_test, axis=0)

        preds_train = batch_downsample(preds_train)
        with open(make_output_path("predicts/train.pkl"), "wb") as f:
            pickle.dump(preds_train, f)

    preds_valid = batch_downsample(preds_valid)
    preds_test = batch_downsample(preds_test)
    generate_submission(preds_valid, ground_truth_valid, preds_test)
