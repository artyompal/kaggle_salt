import os, json, sys, copy, subprocess
import numpy as np
from shutil import copyfile
from torchbearer.callbacks.checkpointers import Best
from torchbearer.callbacks.early_stopping import EarlyStopping
from torchbearer.callbacks.torch_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from .metrics import old_iou_numpy


class Trainer():
    def __init__(self, save_path, code_path, dataset, model_name, model_load_fn, model_new_fn, train_fn):
        self.dataset = dataset
        self.model_name = model_name
        folder_name = model_name.split('_--_')[0].replace('st%i_', '').replace('fold-%i', 'folds')
        self.folder_path = save_path + folder_name + '/'
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
        os.environ['TENSORBOARD_LOGS'] = self.folder_path
        if not model_name.startswith('jupyter'):
            current_script = sys.argv[0]
            copyfile(code_path + current_script, self.folder_path + current_script)
        self.model_load_fn = model_load_fn
        self.model_new_fn = model_new_fn
        self.train_fn = train_fn

    def sequence(self, steps=[], **options):
        latest_i = 0
        for step in steps:
            i = step.pop('step')
            latest_i = i
            if i == 0:
                self.model = self.model_new_fn()
            else:
                model_name = self.find_prev_best_model(i - 1, options.get('fold', None))
                self.model = self.model_load_fn(self.folder_path + model_name)
            opts = copy.deepcopy(options)
            for k, v in step.items():
                opts[k] = v
            self.step(self.model, step_index=i, **opts)
        self.model_name = self.find_prev_best_model(latest_i, options.get('fold', None))
        self.model = self.model_load_fn(self.folder_path + model_name)

    def cross_validate(self, split, test_fn, downsample_fn, **options):
        val_dataset = copy.deepcopy(self.dataset)
        val_dataset.test = split[1]
        val_preds = test_fn(self.model, val_dataset, batch_size=options.get('val_batch', 64))
        val_preds = np.array([downsample_fn(pred) for pred in val_preds.data.cpu().numpy()[:, 0, :, :]])
        val_truth = np.array([self.dataset.mask(record) for record in val_dataset.test])
        assert val_preds.shape == val_truth.shape

        ## Scoring for last model, choose threshold by validation data
        thresholds = np.linspace(-2, 2, 100)
        # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
        #thresholds = np.log(thresholds_ori/(1-thresholds_ori))

        ious = np.array([old_iou_numpy(val_preds, val_truth, prob_thres=threshold) for threshold in thresholds])

        # instead of using default 0 as threshold, use validation data to find the best threshold.
        threshold_best_index = np.argmax(ious)
        iou_best = ious[threshold_best_index]
        threshold_best = thresholds[threshold_best_index]

        val_preds_path = self.folder_path + self.model_name + '_val_preds_iou-%.4f_thres-%.4f'
        val_preds_path = val_preds_path % (iou_best, threshold_best)
        np.save(val_preds_path, val_preds)

        predictions = test_fn(self.model, self.dataset, batch_size=options.get('test_batch', 32))
        predictions = np.array([downsample_fn(pred) for pred in predictions.data.cpu().numpy()[:, 0, :, :]])
        preds_path = self.folder_path + self.model_name + '_test_preds'
        np.save(preds_path, predictions)


    def find_prev_best_model(self, index, fold=None):
        all_files = subprocess.check_output(['ls', self.folder_path]).decode('utf8').split('\n')
        for f in reversed(all_files):
            if f.startswith('st%i_' % index):
                return f
            elif fold is not None and f.startswith('fold-%i_st%i' % (int(fold), index)):
                return f

    def step(self, model, step_index, **options):
        if 'fold' in options.keys():
            model_name = self.model_name % (options['fold'], step_index)
        else:
            model_name = self.model_name % step_index
        save_filepath = self.folder_path + model_name

        print("Train: %s" % save_filepath)
        split = options.pop('split', None)
        if split:
            options['split'] = type(split)
        options['filepath'] = save_filepath
        options['step_index'] = step_index
        options['tensorboard'] = model_name.split('_--')[0]
        options['callbacks'] = ['Best, val_acc, max', 'ReduceLR, val_acc, max, patience 8, factor 0.5', 'Cosine annealing']
        printable_options = "{\n" + "\n".join([("%s %s" % (("'%s':" % str(k)).ljust(12), v)) for k, v in options.items()]) + "\n}"
        with open(self.folder_path + ('params_st%i.json' % step_index), 'w') as fp:
            fp.write(printable_options)
        print("Options:\n%s" % printable_options)
        if split:
            options['split'] = split

        checkpointer = Best(filepath=save_filepath, monitor='val_acc', mode='max')
        cosine = CosineAnnealingLR(25, 1e-6)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode='max', patience=8, factor=0.5, verbose=True)
        early_stop = EarlyStopping(monitor='val_acc', mode='max', patience=50, verbose=True)
        options['callbacks'] = [checkpointer, reduce_lr, cosine, early_stop]

        self.train_fn(model, self.dataset, **options)
