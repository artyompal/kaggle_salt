#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import random
import gc
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook #, tnrange
#from itertools import chain
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
from keras.callbacks import Callback
import keras.backend as K
import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa

import tensorflow as tf
from tta_wrapper import tta_segmentation

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
import imgaug
import time
t_start = time.time()


# In[2]:


VERSION = 32
SEED = 42
FOLDS = 5
DEPTH = True
basic_name = f'Unet_resnet_v{VERSION}'
save_model_name = basic_name + '.model'
save_model_name_lov = basic_name + '_lov.model'
submission_file = basic_name + '.csv'
imgaug.seed(SEED)

print(save_model_name)
print(save_model_name_lov)
print(submission_file)


# In[3]:


img_size_ori = 101
img_size_target = 101

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


# In[4]:


# Loading of training/testing ids and depths
train_df = pd.read_csv("../data/raw/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../data/raw/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)


# In[5]:


train_df["images"] = [np.array(load_img("../data/raw/train/images/{}.png".format(idx),
                                        color_mode = "grayscale",)) / 255 for idx in tqdm_notebook(train_df.index)]


# In[6]:


train_df["masks"] = [np.array(load_img("../data/raw/train/masks/{}.png".format(idx),
                                       color_mode = "grayscale",)) / 255 for idx in tqdm_notebook(train_df.index)]


# In[7]:


train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)


# In[8]:


SUBSET = len(train_df)
train_df = train_df.head(SUBSET)
len(train_df)


# In[9]:


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


# In[10]:


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


# In[11]:


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


# In[12]:


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
        #loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
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
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss


# In[13]:


def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2


# In[14]:


def add_depth_coord(images):
    """ Takes dataset (N, W, H, 1) returns (N, W, H, 3). """
    if not DEPTH:
        return images
    assert(len(images.shape) == 4)
    channel1 = np.zeros_like(images)

    h = images.shape[1]
    for row, const in enumerate(np.linspace(0, 1, h)):
        channel1[:, row, ...] = const

    channel2 = images * channel1
    images = np.concatenate([images, channel1, channel2], axis=-1)
    return images

class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)


# In[15]:


#Data augmentation
import cv2

affine_seq = iaa.Sequential([
# General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-10, 10),
                           translate_percent={"x": (-0.05, 0.05)},
                           mode='edge'),
                # iaa.CropAndPad(percent=((0.0, 0.0), (0.05, 0.0), (0.0, 0.0), (0.05, 0.0)))
                ]),
    # Deformations
    iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.04, 0.08))),
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.1))),
], random_order=True)

intensity_seq = iaa.Sequential([
    iaa.Invert(0.3),
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
                iaa.Add((-10, 10)),
                iaa.AddElementwise((-10, 10)),
                iaa.Multiply((0.95, 1.05)),
                iaa.MultiplyElementwise((0.95, 1.05)),
            ]),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.AverageBlur(k=(2, 5)),
            iaa.MedianBlur(k=(3, 5))
        ])
    ])
], random_order=False)

def augment(x, y):
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip
        sometimes(iaa.Add((-10, 10))),
#        iaa.OneOf([
#            iaa.Noop(),
#            iaa.PerspectiveTransform(scale=(0.04, 0.08)),
#            iaa.Add((-10, 10)),
#            iaa.ContrastNormalization((0.75, 1.5)), 
#            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
#            iaa.EdgeDetect(alpha=(0, 0.7)),
#            iaa.Noop(),
#            sometimes(iaa.OneOf([
#                        iaa.EdgeDetect(alpha=(0, 0.7)),
#                        iaa.DirectedEdgeDetect(
#                            alpha=(0, 0.7), direction=(0.0, 1.0)
#                   ),
#                ])),
#        ]),
        #sometimes(iaa.CropAndPad(
        #        percent=(-0.2, 0.2),
        #        pad_mode=["reflect"]
        #    )),
#        sometimes(iaa.Sequential([
#            iaa.Crop(percent=(0.2), keep_size=False),
#            iaa.Scale({"height": img_size_target, "width": img_size_target}),
#            iaa.Pad(percent=(0.2), pad_mode=["reflect"])
#        ])),
        
    ])._to_deterministic()
    images_aug_x = seq.augment_images(x)
    images_aug_y = seq.augment_images(y)
    return np.array(images_aug_x), np.array(images_aug_y)


# Return augmented images/masks arrays of batch size
def generator(features, labels, batch_size, repeat=1):
    # create empty arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, img_size_target, img_size_target, features.shape[3]))
    batch_labels = np.zeros((batch_size, img_size_target, img_size_target, labels.shape[3]))
    print(batch_features.shape)
    while True:
        # Fill arrays of batch size with augmented data taken randomly from full passed arrays
        indexes = random.sample(range(len(features)), batch_size)*repeat
        # Perform the exactly the same augmentation for X and y
        random_augmented_images, random_augmented_labels =             augment(np.apply_along_axis(np.squeeze, 1, features[indexes]*255).astype(np.uint8),
                    np.apply_along_axis(np.squeeze, 1, labels[indexes]*255).astype(np.uint8))

        yield add_depth_coord(random_augmented_images/255), random_augmented_labels/255
        
#x_train = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 3)
#y_train = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 3) 
#x_test= np.array(test_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 3)

x_train = np.array(train_df.images.tolist()).reshape(-1, img_size_target, img_size_target, 1)
y_train = np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1)
train_cls = np.array(train_df.coverage_class)
gc.collect()
#x_train, y_train, train_cls = augment(train_df)


# In[16]:


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


# In[17]:


"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[18]:


x_test = np.array(
    [(np.array(
        load_img("../data/raw/test/images/{}.png".format(idx),
                 color_mode = "grayscale",))) / 255 for idx in tqdm_notebook(test_df.index)]).reshape(
                        -1, img_size_target, img_size_target, 1)


# In[19]:


from sklearn.model_selection import StratifiedKFold, KFold
def get_adv_cv(data, adv_class=None, folds=FOLDS):
    if len(adv_class)>0:
        print(len(data),len(adv_class))
        assert len(data) == len(adv_class)
        kfold_selector = StratifiedKFold(n_splits=folds, random_state=SEED, shuffle=True)
        return [(train_idx, val_idx) for train_idx, val_idx in kfold_selector.split(data, adv_class)]
    else:
        folds = KFold(n_splits=folds, shuffle=True, random_state=SEED)
        return folds.split(data)

def filter_xy(x, y, th=10): #32
    y = np.array([img if np.sum(img) > 100 else np.zeros_like(img) for img in y])
    y_s = np.array([i.sum() for i in y])
    return x[(y_s==0) | (y_s>th)], y[(y_s==0) | (y_s>th)]


# In[20]:


metric = 'my_iou_metric'
val_metric = 'val_' + metric
restore_from_file = True

metric_lov = 'my_iou_metric_2'
val_metric_lov = 'val_' + metric_lov

early_stopping = EarlyStopping(monitor=val_metric, mode='max', patience=20, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor=val_metric, mode='max', factor=0.25, patience=10,
                              min_lr=0.0001, verbose=1)

early_stopping_lov = EarlyStopping(monitor=val_metric_lov, mode='max', patience=20, verbose=1)

reduce_lr_lov = ReduceLROnPlateau(monitor=val_metric_lov, mode='max', factor=0.25, patience=10,
                                  min_lr=0.00005, verbose=1)

epochs = 400
batch_size = 128
#optimizer = RMSprop(lr=0.0001)
train_cls = np.array(train_df.coverage_class)

def get_oof(x_train, y, x_test):
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    history = {}
    threshold_best = {}
    
    oof_train = np.zeros((ntrain, img_size_ori, img_size_ori))
    oof_test = np.zeros((ntest, img_size_ori, img_size_ori))
    oof_test_skf = np.empty((FOLDS, ntest, img_size_ori, img_size_ori))
    model = None
    
    for i, (train_index, test_index) in enumerate(get_adv_cv(x_train, train_cls, FOLDS)):
        gc.collect()
        print('\nFold {}'.format(i))  
        file_name = "../models/keras_unet_resnet_{0}_f{1}_{2}_v{3}.model".format(SEED, FOLDS, i, VERSION)
        print(file_name)
        y_valid_ori = np.array([y[idx] for idx in test_index])

        model_checkpoint = ModelCheckpoint(file_name,  monitor=val_metric, mode='max',
                                           save_best_only=True, verbose=1)   
        
        x_tr = x_train[train_index, :]
        y_tr = y[train_index]
        x_te = add_depth_coord(x_train[test_index, :])
        y_te = y[test_index]
        print(x_tr.shape, y_tr.shape, x_te.shape)
        x_tr, y_tr = filter_xy(x_tr, y_tr)
        print(x_tr.shape, y_tr.shape, x_te.shape)

        
        x_te_ext = np.append(x_te, [np.fliplr(x) for x in x_te], axis=0)
        y_te_ext = np.append(y_te, [np.fliplr(x) for x in y_te], axis=0)
        #g = generator(x_te, y_te, x_te.shape[0], 4)
        #x_te, y_te = next(g)
        print('new validation size:', x_te_ext.shape, y_te_ext.shape)

        learning_rate = 0.01
        depth = 1
        if DEPTH:
            depth = 3
        input_layer = Input((img_size_target, img_size_target, depth))
        output_layer = build_model(input_layer, 16, 0.5)

        model1 = Model(input_layer, output_layer)
        c = optimizers.adam(lr = learning_rate)
        model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])
        
        
        if (not restore_from_file) or (not os.path.isfile(file_name)):           
            history[(i, 0)] = model1.fit_generator(generator(x_tr, y_tr, batch_size),
                                                   validation_data=[x_te_ext, y_te_ext], 
                                                   epochs=epochs,
                                                   callbacks=[early_stopping, model_checkpoint, reduce_lr],
                                                   use_multiprocessing=True,
                                                   workers=1,
                                                   steps_per_epoch=len(x_tr)*2/batch_size,
                                                  )          

            
                    
            model_lov = load_model(file_name, custom_objects={metric: my_iou_metric})
            input_x = model_lov.layers[0].input
            output_layer = model_lov.layers[-1].input
            model = Model(input_x, output_layer)
            learning_rate = 0.005
            c = optimizers.adam(lr = learning_rate)
            model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])
    
            model_checkpoint = ModelCheckpoint(file_name,
                                               monitor=val_metric_lov,
                                               mode='max',
                                               save_best_only=True,
                                               verbose=1)

            history[(i, 1)] = model.fit_generator(generator(x_tr, y_tr, batch_size),
                                                  validation_data=[x_te_ext, y_te_ext], 
                                                  epochs=epochs,
                                                  callbacks=[early_stopping_lov, model_checkpoint, reduce_lr_lov],
                                                  use_multiprocessing=True,
                                                  workers=1,
                                                  steps_per_epoch=len(x_tr)*2/batch_size,
                                                 )              
            schedule = SGDRScheduler(min_lr=1e-8, max_lr=3e-2, steps_per_epoch=np.ceil(len(x_tr)*2/batch_size),
                        lr_decay=0.8, cycle_length=5, mult_factor=1.5)            
            history[(i, 2)] = model.fit_generator(generator(x_tr, y_tr, batch_size),
                                                  validation_data=[x_te_ext, y_te_ext], 
                                                  epochs=epochs,
                                                  callbacks=[early_stopping_lov, model_checkpoint, schedule],
                                                  use_multiprocessing=True,
                                                  workers=1,
                                                  steps_per_epoch=len(x_tr)*2/batch_size,
                                                 )             
        else:
            model = load_model(file_name, custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                         'lovasz_loss':lovasz_loss})

        
        #tta_model = model#tta_segmentation(model, h_flip=True, merge='mean')
        #tta_model = TTA_ModelWrapper(model)
        
        oof_train[test_index] =             np.array([x for x in predict_result(model, x_te, img_size_target).reshape(-1, img_size_target, img_size_target)])
        oof_test_skf[i, :] =             np.array([x for x in predict_result(model, add_depth_coord(x_test), img_size_target).reshape(-1, img_size_target, img_size_target)])


        thresholds = np.linspace(1e-5, .9999, 50)
        thresholds = np.log(thresholds/(1-thresholds))
        print(thresholds)
        ious = np.array([
            iou_metric_batch(
                y_valid_ori, np.int32(
                    oof_train[test_index] > threshold)) for threshold in tqdm_notebook(thresholds)])        

        threshold_best_index = np.argmax(ious)
        print('ious: ', ious)
        iou_best = ious[threshold_best_index]
        threshold_best[i] = thresholds[threshold_best_index]
        print('threshold_best[{0}]: {1}'.format(i, threshold_best[i]))
        
        print('iou_best: ', iou_best)        
        
        oof_train[test_index] = oof_train[test_index] > threshold_best[i]
        oof_test_skf[i, :] = oof_test_skf[i, :] > threshold_best[i]
        oof_test[:] += oof_test_skf[i, :] / FOLDS
        
        del model
        #del tta_model
        
    #oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test, oof_test_skf, history, threshold_best 


# In[21]:


oof_train, oof_test, oof_test_skf, k_history, threshold_best = get_oof(x_train, y_train, x_test)
#0.802860696517413 0.8163545568039949 0.8210000000000001 0.814142678347935 0.8138190954773868


# In[ ]:


pred_dict = {idx: rle_encode(np.round(oof_test[i])) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}


# In[ ]:


sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
#sub.to_csv('../submissions/submission_oof_{0}_unet_resnet_v{1}.csv'.format(FOLDS, VERSION))


# In[ ]:


val_iou = iou_metric_batch(y_train, oof_train)
val_iou


# In[ ]:


str(np.round(val_iou, 3))[2:]


# In[ ]:


#0.8114000000000001 0.813625 0.8077750000000001


# In[ ]:


gc.collect()
pickle.dump(oof_train, open('../pickle/train_oof_{0}_unet_v{1}'.format(FOLDS, VERSION), 'wb+'), protocol=4)
pickle.dump(oof_test, open('../pickle/test_oof_{0}_unet_v{1}'.format(FOLDS, VERSION), 'wb+'), protocol=4)
#pickle.dump(oof_test_skf, open('../pickle/test_skf_{0}_oof_unet_v{1}'.format(FOLDS, VERSION), 'wb+'), protocol=4)
pickle.dump(threshold_best, open('../pickle/threshold_best_{0}_unet_v{1}'.format(FOLDS, VERSION), 'wb+'), protocol=4)

#for i in oof_test_skf:
    
#pickle.dump(oof_test_skf, open('../pickle/test_skf_{0}_oof_unet_v{1}'.format(FOLDS, VERSION), 'wb+'), protocol=4)


# In[ ]:


valid_dict = {idx: rle_encode(np.round(oof_train[i])) for i, idx in enumerate(tqdm_notebook(train_df.index.values))}
val = pd.DataFrame.from_dict(valid_dict, orient='index')
val.index.names = ['id']
val.columns = ['rle_mask']
val.to_csv('../submissions/oof/train_oof_{0}_unet_resnet_v{1}_ls{2}.csv'.format(FOLDS, VERSION, str(np.round(val_iou, 3))[2:]))
sub.to_csv('../submissions/oof/test_oof_{0}_unet_resnet_v{1}_ls{2}.csv'.format(FOLDS, VERSION, str(np.round(val_iou, 3))[2:]))


# In[ ]:




