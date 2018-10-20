#!/usr/bin/python3.6

##### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import optimizers
from keras.applications.imagenet_utils import preprocess_input


SEED = 42
VERSION = 1

image_size = 197


# Loading of training/testing ids and depths
train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = pd.DataFrame(index=depths_df[~depths_df.index.isin(train_df.index)].index)
test_df = test_df.join(depths_df)

len(train_df)


train_df["images"] = [np.array(load_img("../data/train/images/{}.png".format(idx), interpolation='nearest',
                                        target_size=(image_size, image_size),
                                        color_mode = "grayscale",)) for idx in tqdm_notebook(train_df.index)]

train_df["masks"] = [np.array(load_img("../data/train/masks/{}.png".format(idx), interpolation='nearest',
                                        target_size=(image_size, image_size),
                                       color_mode = "grayscale",)) for idx in tqdm_notebook(train_df.index)]

test_df["images"] = [np.array(load_img("../data/test/images/{}.png".format(idx), interpolation='nearest',
                                        target_size=(image_size, image_size),
                                       color_mode = "grayscale")) for idx in tqdm_notebook(test_df.index)]

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
    images = np.concatenate([images, channel1, channel2], axis=-1)
    return images


x_train = np.array(train_df.images.tolist()).reshape(-1, image_size, image_size, 1)
#x_train /= 255
y_train = np.array(train_df.masks.tolist()).reshape(-1, image_size, image_size, 1)
x_test = np.array(test_df.images.tolist()).reshape(-1, image_size, image_size, 1)
#x_test /= 255
train_cls = np.array(train_df.coverage_class)


ids_train, ids_valid, x_tr, x_val, y_tr, y_val, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    x_train,
    y_train,
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state= SEED)


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


save_model_name = f'../output/resnet50_class_v{VERSION}.model'
early_stopping = EarlyStopping(monitor='val_acc', mode = 'max', patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name, monitor='val_acc',
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode = 'max', factor=0.5, patience=5, min_lr=0.0001, verbose=1)

epochs = 400
batch_size = 32
history = model.fit(preprocess_input(add_depth_coord(x_tr)), y_tr,
                    validation_data=[preprocess_input(add_depth_coord(x_val)), y_val],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint,reduce_lr],
                    verbose=2)

classes_df = pd.DataFrame(index=test_df.index)
classes_df['class'] = model.predict(preprocess_input(add_depth_coord(x_test)))
test_df.to_csv(f'../output/resnet50_class_v{VERSION}.csv', index=True)

