# Image augmentations
from skimage.transform import resize
import albumentations as aug
import numpy as np

SCALED_HEIGHT   = 128
SCALED_WIDTH    = 128
ORIGINAL_HEIGHT = 101
ORIGINAL_WIDTH  = 101

pad128 = aug.PadIfNeeded(min_height=SCALED_HEIGHT, min_width=SCALED_WIDTH)
crop101 = aug.CenterCrop(height=ORIGINAL_HEIGHT, width=ORIGINAL_WIDTH)
upsize128 = aug.Resize(height=SCALED_HEIGHT, width=SCALED_WIDTH)
downsize101 = aug.Resize(height=ORIGINAL_HEIGHT, width=ORIGINAL_WIDTH)
upsize202 = aug.Resize(height=202, width=202)
upsize102 = aug.Resize(height=102, width=102)
crop102 = aug.CenterCrop(height=102, width=102)
pad256 = aug.PadIfNeeded(min_height=256, min_width=256)
crop202 = aug.CenterCrop(height=202, width=202)

STANDARD_MEAN_CHANNEL_0 = 0.485
STANDARD_STD_CHANNEL_0 = 0.229
normalize = aug.Normalize(mean=(STANDARD_MEAN_CHANNEL_0), std=(STANDARD_STD_CHANNEL_0))

class Resizers():
    def __init__(self, aug_type):
        self.aug_type = aug_type

    def downsampler(self, img):
        switcher = {
            'albu': self.downsample_albu,
            'pad_crop': self.downsample_crop,
            'skimage': self.downsample_skimage,
            '256': self.downsample_256,
            'resize_pad_crop': self.downsample_rpc
        }
        return switcher[self.aug_type](img)

    def upsampler(self, img):
        switcher = {
            'albu': self.upsample_albu,
            'pad_crop': self.upsample_pad,
            'skimage': self.upsample_skimage,
            '256': self.upsample_256,
            'resize_pad_crop': self.upsample_rpc
        }
        return switcher[self.aug_type](img)

    def upsample_256(self, img):
        img = upsize202(image=img)['image']
        return pad256(image=img)['image']

    def downsample_256(self, img):
        img = crop202(image=img)['image']
        return downsize101(image=img)['image']

    def upsample_albu(self, img):
        return upsize128(image=img)['image']

    def downsample_albu(self, img):
        return downsize101(image=img)['image']

    def upsample_pad(self, img):
        return pad128(image=img)['image']

    def upsample_rpc(self, img):
        img = upsize102(image=img)['image']
        return pad128(image=img)['image']

    def downsample_crop(self, img):
        return crop101(image=img)['image']

    def downsample_rpc(self, img):
        img = crop102(image=img)['image']
        return downsize101(image=img)['image']

    def upsample_skimage(self, img):
        return resize(img, (SCALED_HEIGHT, SCALED_WIDTH), mode='constant', preserve_range=True)

    def downsample_skimage(self, img):
        return resize(img, (ORIGINAL_HEIGHT, ORIGINAL_WIDTH), mode='constant', preserve_range=True)


class Processors():
    def __init__(self, aug_type):
        self.resizers = Resizers(aug_type)

    def pre(self):
        pre = {
            'image': self.img_preprocess,
            'mask': self.mask_preprocess
        }
        return pre

    def pre_unnormalized(self):
        pre = {
            'image': self.img_preprocess_unnormalized,
            'mask': self.mask_preprocess
        }
        return pre

    def post(self):
        post = {
            'image': self.img_postprocess,
            'mask': self.mask_postprocess
        }
        return post

    def img_preprocess(self, img):
        img = self.resizers.upsampler(img)
        img = img[:,:,0:1]
        img = normalize(image=img)['image']
        return img

    def img_preprocess_unnormalized(self, img):
        img = self.resizers.upsampler(img)
        img = img[:,:,0:1] / 255.
        return img

    def mask_preprocess(self, mask):
        mask = self.resizers.upsampler(mask)
        mask = (mask > 0).astype('float32')
        if mask.sum() < 150:
            mask = np.zeros((mask.shape)).astype('float32')
        return mask

    def img_postprocess(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        return np.transpose(img, (2, 0, 1))

    def mask_postprocess(self, mask):
        return np.expand_dims(mask, 0)
