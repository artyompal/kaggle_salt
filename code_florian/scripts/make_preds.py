import os
import pickle
import sys
import time
import math
import argparse

from typing import Any, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision as vsn

from skimage.io import imread
from tqdm import tqdm

from skimage.transform import resize

from models.nets import ResUNet
from utils.data_loaders import get_data_loaders, get_test_loader
from utils.data_vis import plot_from_torch

from utils.evaluations import DiceLoss, calc_metric, get_iou_vector

import re

# important: parse fold number from filename to avoid typos
def parse_fold_number(filename: str) -> int:
    match = re.match(r'.*_fold-(\d)\.pth', filename)
    if match:
        fold_num = int(match.group(1))
        print("detected fold", fold_num)
    else:
        assert(False)
    return fold_num

parser = argparse.ArgumentParser(description='Make Preds')
parser.add_argument('--imsize', default=128, type=int,
                    help='imsize to use for training')
parser.add_argument('--batch_size', default=128, type=int,
                    help='size of batches')
parser.add_argument('--gpu', default=0, type=int,
                    help='which gpu to run')
# parser.add_argument('--fold_num', type=int, required=True, #default=0,
#                     help='which fold to make predictions for')
parser.add_argument('--weight_file', default='resunet.pth', type=str,
                    help='which weight file to make predictions for')
parser.add_argument('--num_folds', default=5, type=int,
                    help='number of cross val folds')
#parser.add_argument('--model_name', default='resunet', type=str,
 #                   help='name of model for saving/loading weights')
#parser.add_argument('--exp_name', default='tgs_slt', type=str,
#                    help='name of experiment for saving files')
parser.add_argument('--debug', action='store_true',
                    help='whether to display debug info')
parser.add_argument('--flip_tta', action='store_true',
                    help='whether to horizontal flip TTA')
#parser.add_argument('--use_mt', action='store_true',
#                    help='whether to use mean teacher model')
#parser.add_argument('--use_swa', action='store_true',
#                    help='whether to use mean teacher model')
parser.add_argument('--use_bool', action='store_true',
                    help='whether to use empty predictions')
parser.add_argument('--save_raw', action='store_true',
                    help='whether to export predicts without thresholds')
parser.add_argument('--mosaic', default=0, type=int,
                    help='how to use mosaic: 0-disabled, 1-channel 1, 2 - channel 2')
parser.add_argument('--score_only', action='store_true',
                    help='don\'t generate predictions')


def predict(net: Any, test_loader: Any, fold_num: int, to_csv: bool, threshold: float) -> Any:
    net.eval()

    all_predicts = []
    all_masks = []
    rles = []
    ids = []

    # no gradients during validation
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            test_imgs = data['img'].to(device)
            test_ids = data['id']
            blanks = data['blank']

            # get predictions
            preds, chck_preds, edges_preds = net(test_imgs)
            preds = preds.sigmoid()
            chck_preds = chck_preds.sigmoid() > 0.5

            if args.flip_tta:
                test_imgs_lr = data['img_lr'].to(device)
                preds_lr, check_lr, edges_preds_lr = net(test_imgs_lr)
                preds_lr_ = preds_lr.sigmoid()
                check_lr = check_lr.sigmoid() > 0.5

                chck_preds = (check_lr + chck_preds) / 2.
                preds_lr = np.zeros((preds_lr_.size())).astype(np.float32)
                # preds_lr = np.copy(preds_lr_.data.cpu().numpy()[:,:,:,::-1])
                preds_lr = np.copy(preds_lr_.data.cpu().numpy()[:,:,:,::-1])
                # print(preds_lr.shape)

                preds = (preds + torch.from_numpy(preds_lr).to(device)) / 2.

            # set masks to 0 with low probability of having mask
            if args.use_bool:
                chck_preds = chck_preds > 0.5
                preds *= chck_preds.view(chck_preds.size(0),1,1,1).expand_as(preds).float()
            preds *= blanks.view(blanks.size(0),1,1,1).expand_as(preds).float().to(device)

            if args.debug and i == 0:
                img_grid = vsn.utils.make_grid(test_imgs, normalize=True)
                msk_grid = vsn.utils.make_grid(preds)

                if args.flip_tta:
                    img_lr_grid = vsn.utils.make_grid(test_imgs_lr, normalize=True)
                    vsn.utils.save_image(img_lr_grid, '../imgs/test_imgs_lr.png')

                vsn.utils.save_image(img_grid, '../imgs/test_imgs.png')
                vsn.utils.save_image(msk_grid, '../imgs/test_pred.png')

            pred_np = preds.data.cpu().numpy()
            pred_np = pred_np.reshape((-1, pred_np.shape[2], pred_np.shape[3]))

            for j in range(pred_np.shape[0]):
                if args.imsize == 256:
                    predicted_mask = resize(pred_np[j][27:229, 27:229], (101,101),
                                            preserve_range=True)
                else:
                    predicted_mask = pred_np[j][13:114, 13:114]

                ids.append(test_ids[j])

                if to_csv:
                    predicted_mask = np.where(predicted_mask > threshold, 1, 0)
                    rles.append(rle_encode(predicted_mask.astype(np.int32)))
                else:
                    all_predicts.append(predicted_mask)

                    if 'msk' in data:
                        masks = data['msk'].cpu().numpy()
                        masks = masks.reshape(-1, masks.shape[2], masks.shape[3])
                        all_masks.append(masks[j, 13:114, 13:114])

    return (ids, rles) if to_csv else (ids, np.array(all_predicts), np.array(all_masks))

def valid(net, valid_loader, fold_num, use_lovasz=False, save_imgs=False):
    pred = predict(net, valid_loader, fold_num, to_csv=False, threshold=0.4)
    return pred

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

def iou_metric_batch(y_true_in, y_pred_in, threshold):
    # print("iou_metric_batch: y_true_in", y_true_in.shape, "y_pred_in", y_pred_in.shape)
    batch_size = y_true_in.shape[0]
    metric = []

    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch] > threshold)
        metric.append(value)

    return np.mean(metric)

def load_masks(fold_num):
    train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
    # depths_df = pd.read_csv("../data/depths.csv", index_col="id")
    # train_df = train_df.join(depths_df)
    # test_df = depths_df[~depths_df.index.isin(train_df.index)]
    print("train_df", train_df.shape)
    assert(train_df.shape[0] == 4000)

    print("reading train masks")
    masks = np.array([imread("../data/train/masks/%s.png" % idx)
                      for idx in train_df.index])
    masks = np.expand_dims(masks, axis=-1)
    print("masks range:", np.amin(masks), np.amax(masks))
    masks = masks.astype(float) / np.amax(masks)
    masks = np.squeeze(masks)
    print("masks", masks.dtype)

    with open("../data/fixed_folds.pkl", "rb") as f:
        splits = pickle.load(f)

    train_idx, valid_idx = splits[fold_num]
    masks = masks[valid_idx]

    print("masks", masks.shape)
    return masks

def rle_encode(im: Any) -> str:
    '''
    im: numpy array, 1-mask, 0-background
    Returns run length as string
    '''
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def write_csv(filename: str, ids: List[str], rles: List[str]):
    subm = pd.DataFrame.from_dict({'id':ids, 'rle_mask':rles}, orient='index').T
    #if args.use_mt:
    #    subm.to_csv('../subm/{}_{}_mt_fold-{}.csv'.format(args.model_name, args.exp_name, fold_num), index=False)
    #elif args.use_swa:
    #    subm.to_csv('../subm/{}_{}_swa_fold-{}.csv'.format(args.model_name, args.exp_name, fold_num), index=False)
    #else:
    #    subm.to_csv('../subm/{}_{}_best_fold-{}.csv'.format(args.model_name, args.exp_name, fold_num), index=False)
    subm.to_csv(filename, index=False)

    subm.index.names = ['id']
    subm.columns = ['id', 'rle_mask']
    print(subm.head())

def make_preds():
    _, valid_loader = get_data_loaders(imsize=args.imsize,
                                       batch_size=args.batch_size,
                                       num_folds=args.num_folds,
                                       mosaic_mode=args.mosaic,
                                       fold=fold_num)
    print("predicting on the validation dataset")
    ids, preds_val, masks = valid(net, valid_loader, fold_num)

    # TODO: scipy.optimize.minimize
    print("searching for the best threshold")
    thresholds = np.linspace(0.3, 0.7, 31)
    ious = np.array([get_iou_vector(masks, preds_val, threshold) for threshold in tqdm(thresholds)])
    print("iou", ious)

    threshold_best_index = np.argmax(ious)
    best_iou = ious[threshold_best_index]
    best_threshold = thresholds[threshold_best_index]
    print("validation:", best_iou, "best threshold", best_threshold)

    if args.score_only:
        return

    # write predicts for the train set
    directory, name_ext = os.path.split(MODEL_CKPT)
    name, ext = os.path.splitext(name_ext)

    train_predicts = os.path.join(directory, "loc%.04f_train_" % best_iou + name +
                             (".pkl" if args.save_raw else ".csv" ))
    print('generating train predictions to %s' % train_predicts)
    rles = []

    if args.save_raw:
        with open(train_predicts, "wb") as f:
            pickle.dump(preds_val, f)
    else:
        for pred in preds_val:
            pred = np.where(pred > best_threshold, 1, 0)
            rles.append(rle_encode(pred.astype(np.int32)))

        write_csv(train_predicts, ids, rles)

    # write predicts for the test set
    print("predicting on the test dataset")
    test_predicts = os.path.join(directory, "loc%.04f_test_" % best_iou + name +
                                 (".pkl" if args.save_raw else ".csv" ))
    print('generating test predictions to %s' % test_predicts)

    if args.save_raw:
        ids, preds, masks = predict(net, test_loader, fold_num, False, 0)
        print(len(ids), len(preds))
        assert(len(ids) == len(preds))

        with open(test_predicts, "wb") as f:
            pickle.dump((ids, preds), f)
    else:
        ids, preds = predict(net, test_loader, fold_num, True, best_threshold)
        print(len(ids), len(preds))
        assert(len(ids) == len(preds))

        write_csv(test_predicts, ids, preds)

if __name__ == '__main__':
    args = parser.parse_args()
    print("predicting on", args.weight_file)
    fold_num = parse_fold_number(args.weight_file)

    # set model filenames
    #model_params = [args.model_name, args.exp_name, fold_num]
    #if args.use_mt:
    #    MODEL_CKPT = '../model_weights/best_meanteacher_{}_{}_fold-{}.pth'.format(*model_params)
    #elif args.use_swa:
    #    MODEL_CKPT = '../model_weights/swa_{}_{}_fold-{}.pth'.format(*model_params)
    #else:
    #    MODEL_CKPT = '../model_weights/best_{}_{}_fold-{}.pth'.format(*model_params)

    MODEL_CKPT = args.weight_file

    # get the loaders
    test_loader = get_test_loader(imsize=args.imsize, batch_size=args.batch_size,
                                  mosaic_mode=args.mosaic)

    net = ResUNet(use_bool=True)
    if args.gpu == 99:
        device = torch.device("cuda:0")
        net = nn.DataParallel(net, device_ids=[0,1]).cuda()
    else:
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        # torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        net.to(device)

    state_dict = torch.load(MODEL_CKPT, map_location=lambda storage, loc: storage.cuda(args.gpu))
    net.load_state_dict(state_dict)


    make_preds()
