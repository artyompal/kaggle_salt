#!/usr/bin/python3.6

# Input data files are available in the "../data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys
import random
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as albu

import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras import optimizers
from keras.applications.imagenet_utils import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing


SEED = 42
VERSION = 2
BATCH_SIZE = 32
NUM_FOLDS = 5
image_size = 202

PREDICT_ONLY = True


# Loading of training/testing ids and depths
train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = pd.DataFrame(index=depths_df[~depths_df.index.isin(train_df.index)].index)
test_df = test_df.join(depths_df)

len(train_df)


train_df["images"] = [np.array(load_img("../data/train/images/{}.png".format(idx), interpolation='nearest',
                                        target_size=(image_size, image_size),
                                        color_mode = "grayscale",)) for idx in tqdm(train_df.index)]

train_df["masks"] = [np.array(load_img("../data/train/masks/{}.png".format(idx), interpolation='nearest',
                                        target_size=(image_size, image_size),
                                       color_mode = "grayscale",)) for idx in tqdm(train_df.index)]

test_df["images"] = [np.array(load_img("../data/test/images/{}.png".format(idx), interpolation='nearest',
                                        target_size=(image_size, image_size),
                                       color_mode = "grayscale")) for idx in tqdm(test_df.index)]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(image_size, 2) / 255

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i

train_df["coverage_class"] = train_df.coverage.map(cov_to_class)


def get_class(img, th=10):
    img_sum = np.array([i.sum() for i in img])
    return np.array(img_sum>th).astype(int)

def add_depth_coord(images):
    """ Takes dataset (N, W, H, 1) returns (N, W, H, 3). """
    assert(len(images.shape) == 4)
    channel1 = np.zeros_like(images)

    h = images.shape[1]
    for row, const in enumerate(np.linspace(0, 1, h)):
        channel1[:, row, ...] = const

    channel2 = images * channel1
    channel1 *= 255
    images = np.concatenate([images, channel1, channel2], axis=-1)
    return images


x_train = np.array(train_df.images.tolist()).reshape(-1, image_size, image_size, 1)
y_train = np.array(train_df.masks.tolist()).reshape(-1, image_size, image_size, 1)
x_test = np.array(test_df.images.tolist()).reshape(-1, image_size, image_size, 1)
train_cls = np.array(train_df.coverage_class)

class Datagen(keras.utils.Sequence):
    """ Returns batchs of images which are augmented and resized. """
    def __init__(self, x, y, valid):
        assert(x.shape[0] == y.shape[0])
        self.x = x
        self.y = y
        self.valid = valid
        self.preprocessing_fn = get_preprocessing('resnet50')

        SZ = image_size

        self.augs = albu.Compose([
            # albu.OneOf([albu.RandomSizedCrop(min_max_height=(SZ//2, SZ), height=SZ, width=SZ, p=0.5),
            #       albu.PadIfNeeded(min_height=SZ, min_width=SZ, p=0.5)], p=1),
            # albu.VerticalFlip(p=0.5),
            # albu.HorizontalFlip(p=0.5),
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

    def __getitem__(self, idx):
        assert(idx < len(self))

        x = self.x[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
        y = self.y[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]

        if not self.valid:
            xa = []
            for image in x :
                augmented = self.augs(image=image)
                xa.append(augmented["image"].reshape(image_size, image_size, 1))

            x = np.array(xa).reshape(-1, image_size, image_size, 1)

        x = add_depth_coord(x)
        return self.preprocessing_fn(x), y

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / BATCH_SIZE))

folds = StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=666)
pred_train = np.zeros(x_train.shape[0])
pred_test = np.zeros((NUM_FOLDS, x_test.shape[0]))

for fold, indices in enumerate(folds.split(x_train, train_df.coverage_class)):
    print("==================== fold %d" % fold)
    train_idx, valid_idx = indices
    # print("valid_idx", valid_idx.shape)
    x_tr, y_tr = x_train[train_idx], y_train[train_idx]
    x_val, y_val = x_train[valid_idx], y_train[valid_idx]


    #Data augmentation
    x_tr = np.append(x_tr, [np.fliplr(x) for x in x_tr], axis=0)
    y_tr = get_class(np.append(y_tr, [np.fliplr(x) for x in y_tr], axis=0)).flatten()
    y_val = get_class(y_val).flatten()

    resnet_model = ResNet50(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False)
    input_x = resnet_model.input
    output_layer = Flatten()(resnet_model.output)
    output_layer = Dense(1, activation='sigmoid')(output_layer)
    model = Model(input_x, output_layer)
    learning_rate = 0.001
    c = optimizers.adam(lr = learning_rate)
    model.compile(optimizer=c, loss='binary_crossentropy', metrics=['accuracy'])


    save_model_name = '../output/resnet50_class_v%d_fold%d_acc{val_acc:.02f}_epoch{epoch:02d}.model' % (VERSION, fold)
    early_stopping = EarlyStopping(monitor='val_acc', mode = 'max', patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name, monitor='val_acc',
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode = 'max', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

    epochs = 400
    batch_size = 32
    cw = compute_class_weight("balanced", np.unique(y_tr), y_tr)

    if not PREDICT_ONLY:
        model.fit_generator(Datagen(x_tr, y_tr, valid=False),
                            validation_data=Datagen(x_val, y_val, valid=True),
                            epochs=epochs, callbacks=[early_stopping, model_checkpoint, reduce_lr],
                            use_multiprocessing=True, workers=12, shuffle=False,
                            verbose=1, class_weight=cw)

    models = sorted(glob('../output/resnet50_class_v%d_fold%d*.model' % (VERSION, fold)))
    print("models found", models)
    assert(len(models) > 0)
    model = load_model(models[-1])

    pred_test[fold] = np.squeeze(model.predict(preprocess_input(add_depth_coord(x_test)), verbose=1))

    res = model.predict(preprocess_input(add_depth_coord(x_val)), verbose=1)
    pred_train[valid_idx] = np.squeeze(res)

pred_test = np.mean(pred_test, axis=0)

classes_df = pd.DataFrame(index=test_df.index)
classes_df['class'] = pred_test
classes_df.to_csv(f'../output/resnet50_class_test_v{VERSION}.csv', index=True)

classes_df = pd.DataFrame(index=train_df.index)
classes_df['class'] = pred_train
classes_df.to_csv(f'../output/resnet50_class_train_v{VERSION}.csv', index=True)
