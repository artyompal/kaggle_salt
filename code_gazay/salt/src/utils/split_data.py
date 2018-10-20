from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import numpy


def split_data(dataset, random_seed, stratify=True, val_size=0.2):
    split = { 'random_state': random_seed, 'test_size': val_size }
    if dataset.__dict__.get('stratify') and stratify:
        split['stratify'] = [getattr(dataset, dataset.stratify)(record) for record in dataset.train]
    return tuple(train_test_split(dataset.train, **split))

def cross_validate(dataset, **options):
    folds = options.pop('folds', 5)

    if getattr(dataset, 'stratify', None):
        stratify = [getattr(dataset, dataset.stratify)(record) for record in dataset.train]
        f = StratifiedKFold(n_splits=folds, random_state=options.get('random_state'))
        splits = f.split(dataset.train, stratify)
    else:
        f = KFold(n_splits=folds, random_state=options.get('random_state'))
        splits = f.split(dataset.train)

    all_records = numpy.array(dataset.train)
    _splits = []
    for train_ids, valid_ids in splits:
        train_records = all_records[train_ids]
        valid_records = all_records[valid_ids]
        _splits.append((train_records, valid_records))

    return _splits
