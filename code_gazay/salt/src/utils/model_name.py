from time import gmtime, strftime
import subprocess


def id_generator(model_name, with_folds=False):
    ts = strftime("%Y-%m-%d-%H%M%S", gmtime())
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    commit = commit.decode('utf8').replace("\n", '')
    step = 'st%i'
    epoch = '{epoch:02d}'
    val = '{val_loss:.4f}'
    acc = '{val_acc:.4f}'
    _name = '_'.join([step, ts, commit, model_name, '--', epoch, val, acc]) + '.pt'
    if with_folds:
        _name = 'fold-%i_' + _name
    return _name
