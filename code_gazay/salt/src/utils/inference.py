import numpy as np
import torch
from torch.autograd import Variable


class TestBatch(object):
    def _image(self, dataset, img):
        img = dataset.postprocessors['image'](dataset.preprocessors['image'](img))
        img = np.expand_dims(img, 0)
        img = Variable(torch.from_numpy(img)).cuda()
        self.image = img
        return self
