import torch, copy, numpy, os
from .augmentors.image import Augmentor
from .preloader import train_preloader
from .metrics.dice import BCEWithDICELoss, DICELoss
from .metrics.lovasz import LovaszHingeLoss, StableBCELoss, BCELovaszHingeLoss, BCEXLoss
from .metrics.focal import FocalLoss
from torchbearer import Model
from torchbearer.callbacks.tensor_board import TensorBoard
from sklearn.model_selection import KFold, StratifiedKFold


optimizers = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}
losses = {
    'cross_entropy': torch.nn.CrossEntropyLoss,
    'mse': torch.nn.MSELoss,
    'bce': torch.nn.BCEWithLogitsLoss,
    'bce_with_dice': BCEWithDICELoss,
    'dice': DICELoss,
    'lovasz': LovaszHingeLoss,
    'bce_stable': StableBCELoss,
    'bce_xloss': BCEXLoss,
    'bce_lovasz': BCELovaszHingeLoss,
    'focal': FocalLoss
}


def train(net, dataset, **options):
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = options.pop('optimizer', 'adam')

    # We can use optimizer as is if it's initialized optimizer object
    if isinstance(optimizer, tuple) or isinstance(optimizer, str):
        optimizer_name, optimizer_options = _name_with_options(optimizer)
        optimizer = optimizers[optimizer_name](parameters, **optimizer_options)

    loss = options.pop('loss', 'cross_entropy')
    # We can use loss as is if it's initialized loss object
    if isinstance(loss, tuple) or isinstance(loss, str):
        loss_name, loss_options = _name_with_options(loss)
        loss = losses[loss_name](**loss_options)

    epochs = options.pop('epochs', 10)
    batch_size = options.pop('batch_size', 32)
    split = options.pop('split', { 'test_size': 0.25 })
    augmentors = { fields: Augmentor(config) for fields, config in options.pop('augment', {}).items() }

    preload = options.pop('preload', { 'shuffle': True, 'num_workers': 4, 'pin_memory': True })
    preloader = train_preloader(net, dataset, split, batch_size, preload, augmentors)

    callbacks = options.pop('callbacks', []) + [preloader]
    add_tensorboard_callback(callbacks, options.get('tensorboard'))

    metrics = options.pop('metrics', ['acc']) + ['loss']
    device = options.pop('device', 'cuda')

    model = Model(net, optimizer, loss, metrics = metrics).to(device)
    model.train()
    return model.fit_generator(None, epochs=epochs, train_steps=preloader.train_steps, callbacks=callbacks)


def cross_validate(net, dataset, **options):
    nets = []
    folds = options.pop('folds', 3)

    if getattr(dataset, 'stratify', None):
        stratify = [getattr(dataset, dataset.stratify)(record) for record in dataset.train]
        splits = StratifiedKFold(n_splits=folds).split(dataset.train, stratify)
    else:
        splits = KFold(n_splits=folds).split(dataset.train)

    all_records = numpy.array(dataset.train)
    for train_ids, valid_ids in splits:
        train_records = all_records[train_ids]
        valid_records = all_records[valid_ids]

        options['split'] = (train_records, valid_records)
        fold_net = copy.deepcopy(net)
        train(fold_net, dataset, **options)
        nets.append(fold_net)

    return nets


def _name_with_options(obj):
    if isinstance(obj, str):
        return (obj, {})
    else:
        return obj

def add_tensorboard_callback(callbacks, tensorboard_name):
    if tensorboard_name:
        log_dir = os.environ.get('TENSORBOARD_LOGS', './logs')
        cb = TensorBoard(log_dir=log_dir, write_graph=False, comment=tensorboard_name)
        callbacks.append(cb)
