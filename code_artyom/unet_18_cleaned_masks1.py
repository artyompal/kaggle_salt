#!/usr/bin/python3.6

import os, pickle, random, subprocess, sys
from typing import Any

import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model, save_model
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add
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

basic_name = "../output/unet_18_cleaned_masks1"
submission_file = basic_name + '.csv'

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


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation==True: x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3))
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate: x = BatchActivate(x)
    return x


# Build Model
def build_model(input_layer, start_neurons, DropoutRatio=0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons*1, (3,3), activation=None, padding='same')(input_layer)
    conv1 = residual_block(conv1, start_neurons*1)
    conv1 = residual_block(conv1, start_neurons*1, True)
    pool1 = MaxPooling2D((2,2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons*2, (3,3), activation=None, padding='same')(pool1)
    conv2 = residual_block(conv2, start_neurons*2)
    conv2 = residual_block(conv2, start_neurons*2, True)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons*4, (3,3), activation=None, padding='same')(pool2)
    conv3 = residual_block(conv3, start_neurons*4)
    conv3 = residual_block(conv3, start_neurons*4, True)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons*8, (3,3), activation=None, padding='same')(pool3)
    conv4 = residual_block(conv4, start_neurons*8)
    conv4 = residual_block(conv4, start_neurons*8, True)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons*16, (3,3), activation=None, padding='same')(pool4)
    convm = residual_block(convm, start_neurons*16)
    convm = residual_block(convm, start_neurons*16, True)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons*8, (3,3), strides=(2,2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons*8, (3,3), activation=None, padding='same')(uconv4)
    uconv4 = residual_block(uconv4, start_neurons*8)
    uconv4 = residual_block(uconv4, start_neurons*8, True)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons*4, (3,3), strides=(2,2), padding='valid')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons*4, (3,3), activation=None, padding='same')(uconv3)
    uconv3 = residual_block(uconv3, start_neurons*4)
    uconv3 = residual_block(uconv3, start_neurons*4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons*2, (3,3), strides=(2,2), padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons*2, (3,3), activation=None, padding='same')(uconv2)
    uconv2 = residual_block(uconv2, start_neurons*2)
    uconv2 = residual_block(uconv2, start_neurons*2, True)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons*1, (3,3), strides=(2,2), padding='valid')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons*1, (3,3), activation=None, padding='same')(uconv1)
    uconv1 = residual_block(uconv1, start_neurons*1)
    uconv1 = residual_block(uconv1, start_neurons*1, True)

    output_layer_noActi = Conv2D(1, (1,1), padding='same', activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

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

# --------------------------- BINARY LOSSES ---------------------------

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
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss

def train_and_predict(x_train, y_train, x_valid, y_valid, fold):
    # data augmentation
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
    print("x_train after hflip", x_train.shape)
    print("y_train after hflip", y_valid.shape)


    # model
    input_layer = Input((img_size_target, img_size_target, 3))
    output_layer = build_model(input_layer, 17, 0.5)

    model1 = Model(input_layer, output_layer)

    c = optimizers.adam(lr = 0.005)
    model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])

    save_model_name = f"{basic_name}_stage1_fold{fold}.hdf5"

    early_stopping = EarlyStopping(monitor='my_iou_metric', mode = 'max',patience=15, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name, monitor='my_iou_metric', mode='max',
                                       save_best_only=True, verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode='max', factor=0.5, patience=5,
                                  min_lr=0.0001, verbose=1)

    epochs = 80
    batch_size = 128

    if not PREDICT_ONLY:
        history = model1.fit(x_train, y_train,
                             validation_data = [x_valid, y_valid],
                             epochs = epochs,
                             batch_size = batch_size,
                             callbacks = [early_stopping, model_checkpoint, reduce_lr],
                             verbose = 2)


    model1 = load_model(save_model_name, custom_objects={'my_iou_metric':my_iou_metric})
    # remove activation layer and use lovasz loss
    input_x = model1.layers[0].input

    output_layer = model1.layers[-1].input
    model = Model(input_x, output_layer)
    c = optimizers.adam(lr=0.01)

    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

    save_model_name = f"{basic_name}_stage2_fold{fold}.hdf5"

    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=30, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2',
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5,
                                  min_lr=0.00005, verbose=1)
    epochs = 120
    batch_size = 128

    if not PREDICT_ONLY:
        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[ model_checkpoint,reduce_lr,early_stopping],
                            verbose=2)


    model = load_model(save_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                       'lovasz_loss': lovasz_loss})


    def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
        x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
        preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
        preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
        preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
        return preds_test/2


    preds_valid = predict_result(model,x_valid,img_size_target)
    preds_test = predict_result(model,x_test,img_size_target)
    return preds_valid, preds_test


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

def add_depth_coord(images: NpArray) -> NpArray:
    """ Takes dataset (N, W, H, 1) returns (N, W, H, 3). """
    assert(len(images.shape) == 4)
    channel1 = np.zeros_like(images)

    h = images.shape[1]
    for row, const in enumerate(np.linspace(0, 1, h)):
        channel1[:, row, ...] = const

    channel2 = images * channel1
    images = np.concatenate([images, channel1, channel2], axis=-1)
    return images

if __name__ == "__main__":
    enable_logging()
    print(f"training with {NUM_FOLDS} folds")

    train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("../data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    train_df["images"] = [np.array(load_img("../data/train/images/{}.png".format(idx), grayscale=True)) / 255
                          for idx in tqdm(train_df.index)]
    train_df["masks"] = [np.array(load_img("../data/train/masks/{}.png".format(idx), grayscale=True)) / 255
                         for idx in tqdm(train_df.index)]

    if not PREDICT_ONLY:
        train_df["masks"] = train_df["masks"].apply(lambda img: img if np.sum(img) > 150 else np.zeros_like(img))

    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

    images = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    masks = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

    preds_train = np.zeros((train_df.shape[0], img_size_target, img_size_target))
    preds_test = np.zeros((NUM_FOLDS, test_df.shape[0], img_size_target, img_size_target))

    folds = StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=666)
    x_test = np.array([(np.array(load_img("../data/test/images/{}.png".format(idx), grayscale = True))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)

    images = add_depth_coord(images)
    x_test = add_depth_coord(x_test)

    print("train", images.shape)
    print("coverage_class", train_df.coverage_class.shape)
    print("preds_train", preds_train.shape)
    print("preds_test", preds_test.shape)

    for fold, indices in enumerate(folds.split(images, train_df.coverage_class)):
        print("==================== fold %d" % fold)
        train_idx, valid_idx = indices

        x_train, y_train = images[train_idx], masks[train_idx]
        x_valid, y_valid = images[valid_idx], masks[valid_idx]

        p_valid, p_test = train_and_predict(x_train, y_train, x_valid, y_valid, fold)
        preds_train[valid_idx], preds_test[fold] = p_valid, p_test

        with open(make_output_path("predicts/fold%d_test.pkl" % fold), "wb") as f:
            pickle.dump(p_test, f)

        with open(make_output_path("predicts/fold%d_train.pkl" % fold), "wb") as f:
            pickle.dump(p_valid, f)
