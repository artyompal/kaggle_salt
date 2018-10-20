from torch.nn.modules.loss import _WeightedLoss
from torch.nn import BCELoss
from torch.nn import functional as F


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


class DICELoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(DICELoss, self).__init__(weight, size_average, reduce, reduction)


    def forward(self, input, target):
        return dice_loss(input, target)


class BCEWithDICELoss(BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(BCEWithDICELoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input, target):
        bce = F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)
        dice = dice_loss(input, target)
        return bce + dice

# Need to reimplement from Keras from
# here https://www.kaggle.com/alexanderliao/u-net-bn-aug-strat-dice:
# def weighted_dice_loss(y_true, y_pred, weight):
#     smooth = 1.
#     w, m1, m2 = weight, y_true, y_pred
#     intersection = (m1 * m2)
#     score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
#     loss = 1. - K.sum(score)
#     return loss

# def weighted_bce_dice_loss(y_true, y_pred):
#     y_true = K.cast(y_true, 'float32')
#     y_pred = K.cast(y_pred, 'float32')
#     # if we want to get same size of output, kernel size must be odd
#     averaged_mask = K.pool2d(
#             y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
#     weight = K.ones_like(averaged_mask)
#     w0 = K.sum(weight)
#     weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
#     w1 = K.sum(weight)
#     weight *= (w0 / w1)
#     loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
#     return loss
