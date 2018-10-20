import torch, copy
from .augmentors.image import Augmentor
from .preloader import test_preloader
from torchbearer import Model


def ensemble(nets, dataset, **options):
    results = list(map(lambda net: test(net, dataset, **options), nets))
    return torch.stack(results).mean(dim=0)


def test(net, dataset, **options):
    tta = options.pop('tta', 0)
    if tta:
        assert len(options['augment'])
        result = list(map(lambda i: _simple_test(net, dataset, **copy.deepcopy(options)), range(tta)))
        return torch.stack(result).mean(dim=0)
    else:
        return _simple_test(net, dataset, **options)


def _simple_test(net, dataset, **options):
    batch_size = options.pop('batch_size', 32)
    augmentors = { fields: Augmentor(config) for fields, config in options.pop('augment', {}).items() }
    preload = options.pop('preload', { 'shuffle': False, 'num_workers': 4, 'pin_memory': True })
    preloader = test_preloader(net, dataset, batch_size, preload, augmentors)

    callbacks = options.pop('callbacks', []) + [preloader]
    device = options.pop('device', 'cuda')

    model = Model(net, torch.optim.Adam(net.parameters())).to(device)
    model.eval()
    return _forked_predict_generator(model, None, steps=preloader.test_steps, callbacks=callbacks)


# add support for passing callbacks and remove _load_batch_predict
def _forked_predict_generator(model, generator, verbose=2, steps=None, pass_state=False, callbacks=[]):
    import torchbearer
    from torchbearer.callbacks.aggregate_predictions import AggregatePredictions
    from torchbearer.callbacks.callbacks import CallbackList

    state = {
            torchbearer.EPOCH: 0,
            torchbearer.MAX_EPOCHS: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.VALIDATION_GENERATOR: generator
    }
    state.update(model.main_state)

    _callbacks = callbacks + Model._add_printer([AggregatePredictions()], verbose, validation_label_letter='p')

    model._test_loop(state, CallbackList(_callbacks), pass_state, lambda *x: None, steps)

    return state[torchbearer.FINAL_PREDICTIONS]

