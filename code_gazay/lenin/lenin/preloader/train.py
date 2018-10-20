import torchbearer
from .batch import Batch
from .dataloader import dataloader
from torchbearer import Model
from torchbearer.callbacks import Callback
from sklearn.model_selection import train_test_split


def train_preloader(net, dataset, split, batch_size, preload, augmentors):
    if isinstance(split, tuple):
        train_records, valid_records = split
    else:
        if dataset.__dict__.get('stratify'):
            split['stratify'] = [getattr(dataset, dataset.stratify)(record) for record in dataset.train]
        train_records, valid_records = train_test_split(dataset.train, **split)

    train_generator = dataloader(net, dataset, 'train_fields', train_records, batch_size, preload, augmentors)
    # TODO: we need to separate augmentors from preprocessors (that simply prepare input for network)
    valid_generator = dataloader(net, dataset, 'valid_fields', valid_records, batch_size, preload)

    return Batcher(train_generator, valid_generator, dataset.target)


class Batcher(Callback):
    def __init__(self, train_generator, valid_generator, y_true_name):
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.y_true_name = y_true_name

    @property
    def train_steps(self):
        return len(self.train_generator)

    def on_start_training(self, state):
        self.train_iterator = iter(self.train_generator)
        self.valid_iterator = iter(self.valid_generator)

    def on_end_training(self, state):
        state[torchbearer.VALIDATION_GENERATOR] = None
        steps = state[torchbearer.VALIDATION_STEPS] = len(self.valid_iterator)
        model = state[torchbearer.SELF]
        model.eval()
        model._test_loop(state, state[torchbearer.CALLBACK_LIST], False, model._load_batch_none, steps)

    def on_sample(self, state):
        self.batch_from(self.train_iterator, state)

    def on_sample_validation(self, state):
        self.batch_from(self.valid_iterator, state)

    def batch_from(self, iterator, state):
        data = next(iterator)
        records = data.pop('records')
        data = Model._deep_to(data, state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])

        state[torchbearer.X] = Batch(records, data)
        state[torchbearer.Y_TRUE] = data[self.y_true_name]
