# coding: utf-8

SUBM_DIR = '../data/predicts_from_artyom'

CHECKPOINT_PATH = '../model_weights/best_ensemble.pth'
SUBM_NAME = '../subm/ensemble.csv'
GPU = "cuda:0"

NUM_MODELS = 5

import glob, math, sys, os, random, time
import numpy as np, pandas as pd
from tqdm import tqdm
from utils.data_loaders import get_data_loaders
from utils.evaluations import FocalLoss2d, DiceLoss, get_iou_vector
import utils.lovasz_losses as L
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F

def rle_decode(mask_rle, shape=(101, 101)):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    if str(mask_rle) != str(np.nan):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Читает сабмит, сортирует, декодит. Возвращает словарь - id: mask
def read_and_decode(f_path):
    preds_df = pd.read_csv(f_path).sort_values('id')
    ids = []
    preds = {}
    for idx, row in preds_df.iterrows():
        preds[row['id']] = rle_decode(row['rle_mask'])
    assert len(preds_df) == len(preds)
    return preds

# Собираю информацию по всем предиктам в папке - скор, фолд, путь к файлу test/train
best_preds = {}
for f in glob.glob(SUBM_DIR + '/*'):
    f_name = f.split('/')[-1]
    if 'test' in f_name:
        test_f = f
        train_f = f.replace('test', 'train')
    else:
        test_f = f.replace('train', 'test')
        train_f = f
    score = float(f_name.split('_')[0].replace('loc', ''))
    fold = int(f_name.split('-')[-1].replace('.csv', ''))
    if fold in best_preds:
        best_preds[fold].append((score, train_f, test_f))
    else:
        best_preds[fold] = [(score, train_f, test_f)]

# И сразу подгружаю трейн (около 800) и преобразую в словарь чтоб удобней было дальше работать
for fold, v in tqdm(best_preds.items()):
    print("Loading and decoding fold", fold, 'with', len(v), 'snapshots')
    decoded = []
    for vals in sorted(v, key=lambda it: it[0]):
        decoded.append({ 'score': vals[0],
                        'train': read_and_decode(vals[1]),
                        'train_f': vals[1],
                        # Не хватает памяти загрузить сразу и тест для каждого снепшота
                        #'test': read_and_decode(vals[2]),
                        'test_f': vals[2] })
    best_preds[fold] = decoded

# Чото я решил сделать такой адок – чтоб получить настоящие маски использовал и поитерировал по
# валидационному лоадеру для каждого из фолдов -_-
true_masks = {}
for fold in range(5):
    _, valid = get_data_loaders(fold=fold)
    for batch in valid:
        for idx in range(len(batch['id'])):
            true_masks[batch['id'][idx]] = batch['msk'].cpu().numpy()[idx][:,13:114,13:114]

assert len(true_masks) == 4000

# Вот тут я думаю и есть основной проеб. Я так и не понял как связать
# между собой 20% предикта на трейне и все 18к предиктов на тесте.
# Здесь я просто иду по собраной и отсортированной по скору информации
# и собираю "модельки" из последних (то есть лучших по скору)
# предиктов на трейне. А вот с тестом я ничо не понял чо делать -
# они же полные, по 18к. Поэтому я не нашел ничо лучше чем просто
# их усреднить – думаю проблема как раз тут.
def compile_model(best_preds):
    model = {'full':{}}
    for fold in range(5):
        snap = len(best_preds[fold]) - 1
        if len(best_preds[fold]) == 1:
            model[fold] = best_preds[fold][snap]
        else:
            model[fold] = best_preds[fold].pop(snap)
        model['full'].update(model[fold]['train'])
        test = read_and_decode(model[fold]['test_f'])
        if 'test_full' not in model:
            model['test_full'] = test
        else:
            for k, v in test.items():
                model['test_full'][k] = (model['test_full'][k] + test[k])/2.
    assert len(model['full']) == 4000
    assert len(model['test_full']) == 18000
    return model

models = []

for n in range(NUM_MODELS):
    print('Compile model', n)
    models.append(compile_model(best_preds))
assert models[0]['full'].keys() == models[1]['full'].keys()
assert models[0]['test_full'].keys() == models[1]['test_full'].keys()

# Не успел (и думаю что из-за ошибки логики сейчас это не должно сильно влиять)
# поиграться с лоссом. В итоге тренил ловашом
def train(net, optimizer, train_loader, use_lovasz=False):
    iter_loss = 0.

    for i, data in enumerate(train_loader):
        preds = data['preds'].to(device)
        msks = data['msks'].to(device)
        msk_preds = net(preds)
        if use_lovasz:
            loss = L.lovasz_hinge(msk_preds, msks)
        else:
            loss = bce(msk_preds, msks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_loss += loss.item()
        sys.stdout.write('\r')
        sys.stdout.write('B: {:>3}/{:<3} | {:.4}'.format(i+1,
                                            len(train_loader),
                                            loss.item()))

    epoch_loss = iter_loss / (len(train_loader) / batch_size)
    print('\n' + 'Avg Train Loss: {:.4}'.format(epoch_loss))

    return epoch_loss

def valid(net, optimizer, valid_loader, use_lovasz=False):
    net.eval()
    val_ious = []
    val_iter_loss = 0.
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            preds = data['preds'].to(device)
            valid_msks = data['msks'].to(device)
            msk_vpreds = net(preds)

            if use_lovasz:
                vloss = L.lovasz_hinge(msk_vpreds, valid_msks)
            else:
                vloss = bce(msk_vpreds, valid_msks)

            val_iter_loss += vloss.item()
            val_ious.append(get_iou_vector(valid_msks.cpu().numpy(),
                                           msk_vpreds.sigmoid().cpu().numpy()))

    epoch_vloss = val_iter_loss / (len(valid_loader) / batch_size)
    print('Avg Eval Loss: {:.4}, Avg IOU: {:.4}'.format(epoch_vloss, np.mean(val_ious)))
    return epoch_vloss, np.mean(val_ious)

def write_csv(filename, ids, rles):
    subm = pd.DataFrame.from_dict({'id':ids, 'rle_mask':rles}, orient='index').T

    subm.to_csv(filename, index=False)

    subm.index.names = ['id']
    subm.columns = ['id', 'rle_mask']
    print(subm.head())

def predict(net, test_loader, threshold=0.5):
    net.eval()

    all_predicts = []
    all_masks = []
    rles = []
    ids = []

    # no gradients during validation
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            preds = data['preds'].to(device)
            test_ids = data['id']
            preds = net(preds)

            preds = preds.sigmoid()

            pred_np = preds.squeeze().data.cpu().numpy()

            for j in range(pred_np.shape[0]):
                predicted_mask = pred_np[j]

                ids.append(test_ids[j])

                predicted_mask = np.where(predicted_mask > threshold, 1, 0)
                rles.append(rle_encode(predicted_mask.astype(np.int32)))

    return (ids, rles)

def train_network(net, train_loader, valid_loader):
    try:
        # define optimizer
        #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        # С адамом как-то красивей получилось
        optimizer = optim.Adam(net.parameters(), lr=lr)

        train_losses = []
        valid_losses = []
        valid_ious = []

        valid_patience = 0
        best_val_metric = 1000.0
        best_val_iou = 0.0
        use_lovasz = True

        print('Training ...')
        for e in range(epochs):
            print('\n' + 'Epoch {}/{}'.format(e, epochs))

            start = time.time()

            t_l = train(net, optimizer, train_loader, use_lovasz)
            v_l, viou = valid(net, optimizer, valid_loader, use_lovasz)

            if viou > best_val_iou:
                net.eval()
                torch.save(net.state_dict(), CHECKPOINT_PATH)
                best_val_metric = v_l
                best_val_iou = viou
                valid_patience = 0
            else:
                valid_patience += 1

            train_losses.append(t_l)
            valid_losses.append(v_l)
            valid_ious.append(viou)

            print('Time: {}'.format(time.time()-start))

    except KeyboardInterrupt:
        pass

    net.load_state_dict(torch.load(CHECKPOINT_PATH))
    net.eval()

    return (best_val_iou, net)

# В датасете я собираю со всех моделек по предикту для картинки
# сую их в NUM_MODELS слоев
class EnsembleDataset(data.Dataset):
    def __init__(self, models, true_masks, im_ids, valid=False):
        self.models = models
        self.n_models = len(models)
        self.true_masks = true_masks
        self.im_ids = im_ids
        self.valid = valid

    def __getitem__(self, index):
        im_id = self.im_ids[index]
        true_mask = self.true_masks[im_id]
        preds = []
        for n in range(self.n_models):
            preds.append(self.models[n]['full'][im_id])
        preds = np.array(preds)
        return {'id': im_id, 'msks': true_mask, 'preds': preds}

    def __len__(self):
        return len(self.im_ids)

class EnsembleTestDataset(data.Dataset):
    def __init__(self, models, im_ids):
        self.models = models
        self.n_models = len(models)
        self.im_ids = im_ids

    def __getitem__(self, index):
        im_id = self.im_ids[index]
        preds = []
        for n in range(self.n_models):
            preds.append(self.models[n]['test_full'][im_id])
        preds = np.array(preds)
        return {'id': im_id, 'preds': preds}

    def __len__(self):
        return len(self.im_ids)

class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3)):
        super(Conv2dBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Дропаут кстати не юзал
class StackingFCN(nn.Module):
    def __init__(self, input_model_nr, num_classes, filter_nr=32, dropout_2d=0.0):
        super().__init__()
        self.dropout_2d = dropout_2d

        self.conv = nn.Sequential(Conv2dBnRelu(input_model_nr, filter_nr, kernel_size=(3, 3)),)
        self.final = nn.Sequential(nn.Conv2d(filter_nr, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        x = x.float()
        x = F.dropout2d(self.conv(x), p=self.dropout_2d)
        return self.final(x)

bce = nn.BCEWithLogitsLoss()

# Спличу для тренировки трейн сет
im_ids = list(models[0]['full'])
train_ids, val_ids = train_test_split(im_ids, test_size=0.2)

train_dataset = EnsembleDataset(models=models, true_masks=true_masks, im_ids=train_ids)
valid_dataset = EnsembleDataset(models=models, true_masks=true_masks, im_ids=val_ids, valid=True)

BATCH_SIZE = 64

train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

lr = 0.001
epochs = 100

device = torch.device(GPU)
net = StackingFCN(input_model_nr=5, num_classes=1)
net.to(device)
net.train()

best_iou, net = train_network(net, train_loader, valid_loader)

print("Loaded model with best IOU", best_iou)

test_im_ids = list(models[0]['test_full'])
test_dataset = EnsembleTestDataset(models=models, im_ids=test_im_ids)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

ids, rles = predict(net, test_loader)

write_csv(SUBM_NAME, ids, rles)
