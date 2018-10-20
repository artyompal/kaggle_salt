from lenin import train, test, cross_validate, ensemble
from lenin.datasets.xy import XY
from lenin.transforms import hwc_to_chw
import torch, numpy

import torchvision
dataset = XY(torchvision.datasets.CIFAR10(root='../datasets/cifar', train=True, download=True))

from albumentations import Normalize
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset.preprocessors = { 'x': numpy.array }
dataset.postprocessors = { 'x': [normalize, hwc_to_chw] }


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, stride=2, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, stride=2, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, stride=2, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.classifier = torch.nn.Linear(576, 10)

    def forward(self, batch):
        x = batch.x
        x = self.convs(x)
        x = x.view(-1, 576)
        return self.classifier(x)


net = SimpleNet()
train(net, dataset, epochs=1, preload={'num_workers': 0 })

dataset.test = dataset.train[:100]
test(net, dataset, tta=2, augment={ 'x': [{ 'type': 'Flip' }]}, preload={'num_workers': 0})

nets = cross_validate(net, dataset, epochs=1)
ensemble(nets, dataset)
