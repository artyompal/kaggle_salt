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
from glob import glob

import numpy as np, pandas as pd, scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold

from tqdm import tqdm
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing

import keras
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

import albumentations as albu


NpArray = Any

ENABLE_KFOLD = True
EPOCHS      = 1000
BATCH_SIZE  = 32
VERBOSE     = 1
NUM_FOLDS   = 5 if ENABLE_KFOLD else 1

img_size_ori = 101
img_size_target = 224


def enable_logging() -> None:
    """ Sets up logging to a file. """
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    log_file = '../output/' + module_name + ".log"

    tee = subprocess.Popen(["tee", "-a", log_file], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())

def make_output_path(filename: str) -> str:
    """ Returns a correct file path to save to. """
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    name_ext = os.path.splitext(filename)
    return '../output/' + name_ext[0] + '_' + module_name + name_ext[1]

def upsample(img: NpArray) -> NpArray:
    if img.shape[0] == img_size_target and img.shape[1] == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant',
                  preserve_range=True, anti_aliasing=True)

def downsample(img: NpArray) -> NpArray:
    if img.shape[0] == img_size_ori and img.shape[1] == img_size_ori:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant',
                  preserve_range=True, anti_aliasing=True)

def batch_upsample(batch: NpArray) -> NpArray:
    if batch.shape[1] == img_size_target and batch.shape[2] == img_size_target:
        return batch

    res = np.array([resize(img, (img_size_target, img_size_target),
                           mode='constant', preserve_range=True, anti_aliasing=True)
                   for img in batch])
    return res

def batch_downsample(batch: NpArray) -> NpArray:
    if batch.shape[1] == img_size_ori and batch.shape[2] == img_size_ori:
        return batch

    res = np.array([resize(img, (img_size_ori, img_size_ori), mode='constant',
                           preserve_range=True, anti_aliasing=True)
                   for img in batch])
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

class Datagen(keras.utils.Sequence):
    """ Returns batchs of images which are augmented and resized. """
    def __init__(self, x: NpArray, y: NpArray, valid: bool) -> None:
        assert(x.shape[0] == y.shape[0])
        self.x = x
        self.y = y
        self.preprocessing_fn = get_preprocessing('resnet34')

        SZ = img_size_ori

        self.augs = albu.Compose([
            # albu.OneOf([albu.RandomSizedCrop(min_max_height=(SZ//2, SZ), height=SZ, width=SZ, p=0.5),
            #       albu.PadIfNeeded(min_height=SZ, min_width=SZ, p=0.5)], p=1),
            # albu.VerticalFlip(p=0.5),
            albu.HorizontalFlip(p=0.5),
            # albu.RandomRotate90(p=0.5),
            albu.Rotate(p=0.5, limit=10),
            albu.OneOf([
                albu.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                albu.GridDistortion(p=0.5),
                albu.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
                ], p=0.8),
            # albu.CLAHE(p=0.8),
            # albu.RandomContrast(p=0.8),
            albu.RandomBrightness(p=0.8),
            albu.RandomGamma(p=0.8)])

        print("created Datagen: x", x.shape, "y", y.shape)

    def __getitem__(self, idx: int) -> NpArray:
        assert(idx < len(self))

        x = self.x[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
        x = [upsample(img) for img in x]

        y = self.y[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
        y = [upsample(img) for img in y]

        xa, ya = [], []
        for image, mask in zip(x, y):
            augmented = self.augs(image=image, mask=mask)
            xa.append(augmented["image"])
            ya.append(augmented["mask"])

            if len(ya[-1].shape) == 2:
                ya[-1] = np.expand_dims(ya[-1], axis=-1)

        x, y = np.array(xa), np.array(ya)
        x = self.preprocessing_fn(x)
        return x, y

    def __len__(self) -> int:
        return int(np.ceil(self.x.shape[0] / BATCH_SIZE))

def train(x_train: NpArray, x_valid: NpArray, y_train: NpArray, y_valid: NpArray,
          fold: int = -1) -> str:
    names = list(glob("../output/models/fold%d_epoch*_unet_resnet34_01_aug.hdf5" % fold))
    names.sort()
    assert(len(names) > 0)
    model_name = names[-1]
    print("startup model found", model_name)

    model = load_model(model_name, custom_objects={'my_iou_metric': my_iou_metric})

    # remove last activation layer and use Losvasz loss
    input_x = model.layers[0].input
    output_layer = model.layers[-1].input
    model = Model(input_x, output_layer)

    # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
    # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
    opt = optimizers.adam(lr=1e-5)
    model.compile(loss=lovasz_loss, optimizer=opt, metrics=[my_iou_metric_2])

    model_name = make_output_path("models/fold%d_epoch{epoch:02d}_iou{val_my_iou_metric_2:.4f}.hdf5" % fold)
    model_checkpoint = ModelCheckpoint(model_name, monitor='val_my_iou_metric_2',
                                       mode='max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode='max',
                                  factor=0.5, patience=5, min_lr=3e-6, verbose=1)

    model.fit_generator(Datagen(x_train, y_train, valid=False),
                        validation_data=Datagen(x_valid, y_valid, valid=True),
                        epochs=EPOCHS, callbacks=[model_checkpoint, reduce_lr],
                        use_multiprocessing=True, workers=12,
                        shuffle=False, verbose=VERBOSE)

    names = list(glob(make_output_path("models/fold%d_epoch*_iou*.hdf5" % fold)))
    names.sort()
    assert(len(names) > 0)
    print("model found", names[-1])
    return names[-1]

class TestDatagen(keras.utils.Sequence):
    """ Returns batchs of resized images with TTA. """
    def __init__(self, images: NpArray, flip: bool) -> None:
        self.images = images
        self.flip = flip
        print(f"created TestDatagen images={images.shape} flip={flip}")

    def __getitem__(self, idx: int) -> NpArray:
        assert(idx < len(self))

        batch_images = self.images[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
        x = np.array([upsample(np.array(load_img("../data/test/images/%s.png" % img)))
                       / 255 for img in batch_images])
        if self.flip:
            x = np.array([np.fliplr(img) for img in x])

        return x

    def __len__(self) -> int:
        return int(np.ceil(self.images.size / BATCH_SIZE))

def predict_datagen(model: Any, image_list: NpArray) -> NpArray:
    """ Takes a list of file and predicts results with TTA. """
    print("predicting on test set")
    preds_test = model.predict_generator(
        TestDatagen(image_list, flip=False),
        use_multiprocessing=True, workers=12,
        verbose=VERBOSE)
    preds_test = batch_downsample(preds_test)

    print("predicting on flipped test set")
    preds_reflected = model.predict_generator(
        TestDatagen(image_list, flip=True),
        use_multiprocessing=True, workers=12,
        verbose=VERBOSE)
    preds_reflected = batch_downsample(preds_test)

    preds_test += np.array([np.fliplr(x) for x in preds_reflected])
    return preds_test / 2

def predict_result(model: Any, x_test: NpArray) -> NpArray:
    """ Predicts for both orginal and reflected dataset. """
    print("predicting on validation set")
    x_test = batch_upsample(x_test)
    x_test_reflect = np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test, verbose=VERBOSE)
    preds_test = preds_test.reshape(-1, img_size_target, img_size_target)

    print("predicting on flipped validation set")
    preds_test2_reflect = model.predict(x_test_reflect, verbose=VERBOSE)
    preds_test2_reflect = preds_test2_reflect.reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([np.fliplr(x) for x in preds_test2_reflect])
    return preds_test / 2

def predict(model_name: str, x_valid: NpArray, x_test: NpArray) -> Tuple[NpArray, NpArray]:
    model = load_model(model_name,
                       custom_objects={'my_iou_metric': my_iou_metric,
                                       'my_iou_metric_2': my_iou_metric_2,
                                       'lovasz_loss': lovasz_loss})

    print("predicting...")
    print("x_valid", x_valid.shape)
    print("x_test", x_test.shape)

    preds_valid = np.squeeze(batch_downsample(predict_result(model, x_valid)))
    preds_test = np.squeeze(batch_downsample(predict_datagen(model, x_test)))
    return preds_valid, preds_test

def generate_submission(preds_valid: NpArray, ground_truth_valid: NpArray,
                        preds_test: NpArray, testset_images: NpArray) -> None:
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
                 for i, idx in enumerate(tqdm(testset_images))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(make_output_path("sub.csv"))

def load_train_data() -> Tuple[NpArray, NpArray, NpArray, NpArray]:
    train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("../data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    print("train_df", train_df.shape, "test_df", test_df.shape)

    print("reading train images")
    train_df["images"] = [np.array(load_img("../data/train/images/%s.png" % idx))
                          / 255 for idx in tqdm(train_df.index)]
    images = np.array(train_df.images.tolist())
    print("images", images.shape)

    print("reading train masks")
    train_df["masks"] = [np.array(load_img("../data/train/masks/%s.png" % idx,
                                           color_mode="grayscale"))
                         / 255 for idx in tqdm(train_df.index)]
    masks = np.array(train_df.masks.tolist())
    masks = np.expand_dims(masks, axis=-1)
    print("masks", masks.shape)

    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    labels_for_strat = train_df.coverage_class
    print("labels_for_strat", labels_for_strat.shape)

    return images, masks, labels_for_strat, test_df.index.values

if __name__ == "__main__":
    enable_logging()
    if ENABLE_KFOLD:
        print(f"training with {NUM_FOLDS} folds")
    else:
        print("training without folds")

    images, masks, labels_for_strat, testset = load_train_data()

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")
    # print("memory stats:")
    # for stat in top_stats[:10]:
    #     print(stat)

    if not ENABLE_KFOLD:
        x_train, x_valid, y_train, y_valid, = \
            train_test_split(images, masks, stratify=labels_for_strat,
                             shuffle=True, random_state=666, test_size=0.2)

        model_name = train(x_train, x_valid, y_train, y_valid)

        preds_valid, preds_test = predict(model_name, x_valid, testset)
        ground_truth_valid = y_valid
    else:
        preds_train = np.zeros((images.shape[0], img_size_ori, img_size_ori))
        preds_test = np.zeros((NUM_FOLDS, testset.shape[0], img_size_ori, img_size_ori))
        print("preds_train", preds_train.shape)
        print("preds_test", preds_test.shape)

        folds = StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=666)

        for fold, indices in enumerate(folds.split(images, labels_for_strat)):
            print("==================== fold %d" % fold)
            val_filename = make_output_path("predicts/fold%d_val.pkl" % fold)
            test_filename = make_output_path("predicts/fold%d_test.pkl" % fold)
            train_idx, valid_idx = indices

            if os.path.exists(val_filename) and os.path.exists(test_filename):
                with open(val_filename, "rb") as f:
                    p_val = pickle.load(f)
                with open(test_filename, "rb") as f:
                    p_test = pickle.load(f)
            else:
                x_train, y_train = images[train_idx], masks[train_idx]
                x_valid, y_valid = images[valid_idx], masks[valid_idx]

                model_name = train(x_train, x_valid, y_train, y_valid, fold)
                p_val, p_test = predict(model_name, x_valid, testset)

                with open(val_filename, "wb") as f:
                    pickle.dump(p_val, f)
                with open(test_filename, "wb") as f:
                    pickle.dump(p_test, f)

            preds_train[valid_idx] = p_val
            preds_test[fold] = p_test

        preds_valid = preds_train
        ground_truth_valid = masks
        preds_test = np.mean(preds_test, axis=0)

        preds_train = batch_downsample(preds_train)
        with open(make_output_path("predicts/train.pkl"), "wb") as f:
            pickle.dump(preds_train, f)

    preds_valid = batch_downsample(preds_valid)
    preds_test = batch_downsample(preds_test)
    generate_submission(preds_valid, ground_truth_valid, preds_test, testset)
