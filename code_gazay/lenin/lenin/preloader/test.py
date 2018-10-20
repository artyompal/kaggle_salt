import torchbearer
from .batch import Batch
from .dataloader import dataloader
from torchbearer import Model
from torchbearer.callbacks import Callback


def test_preloader(net, dataset, batch_size, preload, augmentors):
    test_generator = dataloader(net, dataset, 'test_fields', dataset.test, batch_size, preload, augmentors)
    return Batcher(test_generator)


class Batcher(Callback):
    def __init__(self, test_generator):
        self.test_iterator = iter(test_generator)

    @property
    def test_steps(self):
        return len(self.test_iterator)

    def on_sample_validation(self, state):
        self.batch_from(self.test_iterator, state)

    def batch_from(self, iterator, state):
        data = next(iterator)
        records = data.pop('records')
        data = Model._deep_to(data, state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
        state[torchbearer.X] = Batch(records, data)
