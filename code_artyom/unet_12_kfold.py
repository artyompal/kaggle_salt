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

import os, pickle, random, re, sys, subprocess
from glob import glob
from typing import *

import numpy as np, pandas as pd, scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold

from tqdm import tqdm
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize

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
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img


NpArray = Any

ENABLE_KFOLD = True
EPOCHS1      = 100
EPOCHS2      = 200
NUM_FOLDS    = 5 if ENABLE_KFOLD else 1

img_size_ori = 101
img_size_target = 101


basic_name = "../output/unet_12_kfold"

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

def upsample(img: NpArray) -> NpArray: # not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

def downsample(img: NpArray) -> NpArray: # not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

def cov_to_class(val: float) -> int:
    return int(val / 0.1)

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

# Build model
def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)

    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)

    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)

    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)

    return output_layer

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
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred >0], tf.float64)

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
        loss = tf.tensordot(tf.nn.elu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
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
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
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

def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2

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

def parse_iou(filename: str) -> float:
    m = re.match(r".+_iou(\d+\.\d*).*", filename)
    return float(m.group(1))

def train(x_train: NpArray, x_valid: NpArray, y_train: NpArray, y_valid: NpArray,
          fold: int = -1) -> None:
    learning_rate = 1e-4
    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = build_model(input_layer, 16,0.5)

    model1 = Model(input_layer, output_layer)

    c = optimizers.adam(lr = learning_rate)
    model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])
    #model1.summary()

    early_stopping = EarlyStopping(monitor='my_iou_metric', mode = 'max',patience=10, verbose=1)
    save_model_name = basic_name + '_stage1_fold%d_epoch{epoch:02d}_iou{my_iou_metric:.4f}.hdf5' % fold

    model_checkpoint = ModelCheckpoint(save_model_name,monitor='my_iou_metric',
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)

    epochs = EPOCHS1
    batch_size = 32
    history = model1.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint,reduce_lr],
                        verbose=2)


    names = list(glob(basic_name + '_stage1*.hdf5'))
    names.sort(key=parse_iou)
    assert(len(names) > 0)
    save_model_name = names[-1]
    print("stage1 model found", save_model_name)


    model1 = load_model(save_model_name,custom_objects={'my_iou_metric': my_iou_metric})
    # remove later activation layer and use losvasz loss
    input_x = model1.layers[0].input

    output_layer = model1.layers[-1].input
    model = Model(input_x, output_layer)
    c = optimizers.adam(lr = learning_rate)

    # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
    # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

    #model.summary()


    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
    save_model_name = basic_name + '_stage2_fold%d_epoch{epoch:02d}_iou{val_my_iou_metric_2:.4f}.hdf5' % fold
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2',
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    epochs = EPOCHS2
    batch_size = 32

    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[ model_checkpoint,reduce_lr,early_stopping],
                        verbose=2)

def predict(x_valid: NpArray, x_test: NpArray, fold: int = -1) -> Tuple[NpArray, NpArray]:
    names = list(glob(basic_name + '_stage2_fold%d_*.hdf5' % fold))
    names.sort(key=parse_iou)
    assert(len(names) > 0)
    save_model_name = names[-1]
    print("stage2 model found", save_model_name)

    model = load_model(save_model_name,
                       custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                       'lovasz_loss': lovasz_loss})

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
    pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best))
                 for i, idx in enumerate(tqdm(test_df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(make_output_path("sub.csv"))

if __name__ == "__main__":
    enable_logging()
    if ENABLE_KFOLD:
        print(f"training with {NUM_FOLDS} folds")
    else:
        print("training without folds")

    # Loading of training/testing ids and depths
    train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("../data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    print("train_df", train_df.shape, "test_df", test_df.shape)

    print("reading train images")
    train_df["images"] = [np.array(load_img("../data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]

    print("reading train masks")
    train_df["masks"] = [np.array(load_img("../data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]

    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

    images = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    masks = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    labels_for_strat = train_df.coverage_class

    print("reading test set")
    x_test = np.array([(np.array(load_img("../data/test/images/{}.png".format(idx), grayscale = True))) / 255
                       for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)

    preds_train = np.zeros((train_df.shape[0], img_size_target, img_size_target))
    preds_test = np.zeros((NUM_FOLDS, test_df.shape[0], img_size_target, img_size_target))

    print("train", images.shape)
    print("labels_for_strat", labels_for_strat.shape)
    print("preds_train", preds_train.shape)
    print("preds_test", preds_test.shape)

    if not ENABLE_KFOLD:
        x_train, x_valid, y_train, y_valid = train_test_split(images,
              masks, stratify=labels_for_strat, shuffle=True, random_state=666)

        train(x_train, x_valid, y_train, y_valid)
        preds_valid, preds_test[0] = predict(x_valid, x_test)
        ground_truth_valid = y_valid
        preds_test = preds_test[0]

        with open(make_output_path("predicts/fold-1_test.pkl"), "wb") as f:
            pickle.dump(preds_test, f)
    else:
        folds = StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=666)

        for fold, indices in enumerate(folds.split(images, labels_for_strat)):
            print("==================== fold %d" % fold)

            train_idx, valid_idx = indices
            x_train, y_train = images[train_idx], masks[train_idx]
            x_valid, y_valid = images[valid_idx], masks[valid_idx]

            # data augmentation
            x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
            y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

            train(x_train, x_valid, y_train, y_valid, fold)

            p_val, p_test = predict(x_valid, x_test, fold)
            preds_train[valid_idx], preds_test[fold] = p_val, p_test

            with open(make_output_path("predicts/fold%d_test.pkl" % fold), "wb") as f:
                pickle.dump(p_test, f)

        preds_valid = preds_train
        ground_truth_valid = masks
        preds_test = np.mean(preds_test, axis=0)

        with open(make_output_path("predicts/train.pkl"), "wb") as f:
            pickle.dump(preds_train, f)

    generate_submission(preds_valid, ground_truth_valid, preds_test)
