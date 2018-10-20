import os
root = os.environ['HOME'] + '/'
competition_name = "tgs-salt-identification-challenge"
short_comp_name = "salt"
DATASET_PATH     = root + "datasets/competitions/" + competition_name
SAVE_PATH        = root + "models/" + short_comp_name + '/'
SUBMISSIONS_PATH = root + "submissions/" + short_comp_name + '/'
SRC_PATH         = root + "code/kaggle_salt/code_gazay/"

COMP_SRC_PATH    = SRC_PATH + short_comp_name
SCRIPTS_PATH     = COMP_SRC_PATH + '/scripts/'
LENIN_SRC_PATH   = SRC_PATH + "lenin"

ARCH_NAME = 'albunet34'

# Debug
from pdb import set_trace

# stdlib
import sys, copy, datetime

# Lenin straight from repo
sys.path.insert(0, LENIN_SRC_PATH)
import lenin
from lenin import train, test
from lenin.datasets.salt import Dataset

# Torch
import torch

# Torchbearer
import torchbearer
from torchbearer import metrics

# Utils from competition directory
sys.path.insert(0, COMP_SRC_PATH)
from src.utils import id_generator
from src.utils.split_data import split_data, cross_validate
from src.utils.randomization import set_random_seed, base_model_name
from src.utils.train_wrapper import Trainer
from src.utils.metrics import intersection_over_union, intersection_over_union_thresholds

START = datetime.datetime.now()

# Randomization (setting random seed and model name)
global RANDOM_SEED
RANDOM_SEED = set_random_seed(4621)
MODEL_NAME = base_model_name(ARCH_NAME)
print(MODEL_NAME)

# Model architecture
from src.models.albunet34 import new_model, load_model

# Metric
@metrics.default_for_key('acc')
@metrics.running_mean
@metrics.std
@metrics.mean
@metrics.lambda_metric('acc', on_epoch=False)
def iou_pytorch(y_pred: torch.Tensor, y_true: torch.Tensor, prob_thres=0):
    batch_size = y_true.shape[0]

    metric = torch.tensor([]).float().cuda()
    for batch in range(batch_size):
        t, p = y_true[batch]>0, y_pred[batch]>0

        intersection = (t & p)
        union = (t | p)
        iou = (torch.sum(intersection > 0).float() + 1e-10 ) / (torch.sum(union > 0).float() + 1e-10)
        thresholds = torch.arange(0.5, 1, 0.05).float().cuda()
        s = torch.tensor([]).float().cuda()
        for thresh in thresholds:
            s = torch.cat((s, (iou > thresh).float().unsqueeze(0)))
        metric = torch.cat((metric, torch.mean(s).unsqueeze(0)))

    return torch.mean(metric)

# Images resizing
from src.utils.processors import Resizers, Processors

# Hyperparams

stratify = True
val_size = 0.08
orig_options = {
    'random_seed': RANDOM_SEED,
    # For debug purposes set num workers to 0
    #'preload': { 'num_workers': 0 }, # 'pin_memory': True, 'worker_init_fn': _init_fn },

    'augment': { ('image', 'mask'): [
        {'type': 'HorizontalFlip'},
        {'type': 'ShiftScaleRotate', 'rotate_limit': 17, 'scale_limit': 0.05, 'shift_limit': 0.05},
        {'type': 'Blur'},
        {'type': 'GaussNoise' },
        #{'type': 'RandomGamma' },
        #{'type': 'OpticalDistortion' },
        #{'type': 'GridDistortion' },
        #{'type': 'ElasticTransform' },
        #{'type': 'HueSaturationValue' },
        #{'type': 'RandomBrightness' },
        #{'type': 'RandomContrast' },
        #{'type': 'MotionBlur' },
        #{'type': 'MedianBlur' },
        #{'type': 'CLAHE' },
        #{'type': 'JpegCompression' },
    ]},
    'batch_size': 96,
    'optimizer': ('adam', { 'lr': 1e-4 }),
    'epochs': 100,
    'loss': 'focal',
    'metrics': ['loss', 'acc'],
    'val_size': val_size,
    'stratify_split': stratify,
}

steps = [
    {
        'step': 0,
    },
    {
        'step': 1,
        'optimizer': ('adam', { 'lr': 1e-5 }),
        'loss': 'lovasz',
        'epochs': 500
    }
]

PROCESSING = 'resize_pad_crop'
orig_options['processing'] = PROCESSING

FOLDS = 5
if len(sys.argv) > 1:
    FOLD = int(sys.argv[1])

# Dataset
dataset = Dataset(DATASET_PATH)
procs = Processors(aug_type=PROCESSING)
dataset.preprocessors = procs.pre()
dataset.postprocessors = procs.post()
downsample_fn = procs.resizers.downsampler

# Trainer
model_name = id_generator(MODEL_NAME, with_folds=(FOLDS > 0))
#model_base = '2018-10-10-045239_1f5e39f_seed-4621_albunet34'
#model_name = 'st%i_' + model_base + '_--_{epoch:02d}_{val_loss:.4f}_{val_acc:.4f}.pt'

trainer = Trainer(save_path=SAVE_PATH,
                  code_path=SCRIPTS_PATH,
                  dataset=dataset,
                  model_name=model_name,
                  model_load_fn=load_model,
                  model_new_fn=new_model,
                  train_fn=train)

if FOLDS > 0:
    validation_split = split_data(dataset, RANDOM_SEED, stratify=stratify, val_size=val_size)
    dataset.train = validation_split[0]
    splits = cross_validate(dataset, folds=FOLDS, random_state=RANDOM_SEED)
    split = splits[FOLD]

    orig_options['split'] = split
    orig_options['fold'] = FOLD
    trainer.sequence(steps, **orig_options)

    trainer.cross_validate(validation_split, test, downsample_fn, val_batch=32, test_batch=32)
else:
    orig_options['split'] = split_data(dataset, RANDOM_SEED, stratify=stratify, val_size=val_size)
    trainer.sequence(steps, **orig_options)
    trainer.cross_validate(orig_options['split'], test, downsample_fn, val_batch=32, test_batch=32)

diff = datetime.datetime.now() - START
minutes = diff.seconds // 60
hours = minutes // 60
seconds = diff.seconds % 60

print("Experiment run for %ih %im %is" % (hours, minutes, seconds))
