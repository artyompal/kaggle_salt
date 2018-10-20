# record: {train,test}-{image id}

# depths.csv:
# image id, depth
#
# train.csv:
# image id, run-length-encoding of mask
#
# {train,test}/images/{image id}.png
# {train,test}/masks/{image id}.png

import numpy as np
from imageio import imread
from pandas import read_csv
from os.path import isfile
from skimage.transform import resize

class Dataset:
    fields = ['image', 'depth']
    target = 'mask'
    stratify = 'coverage_split'
    coverage_splits = 10

    def __init__(self, root):
        self.root = root
        self.depths = read_csv(root + '/depths.csv', index_col=0)['z']
        self.masks = read_csv(root + '/train.csv', index_col=0)['rle_mask']
        self.train = 'train-' + self.masks.index
        self.test = 'test-' + read_csv(root + '/sample_submission.csv', index_col=0).index


    def check_integrity(self):
        for record in self.train:
            assert isfile(self.path(record, 'image')), record
            assert isfile(self.path(record, 'mask')), record
            assert isinstance(self.depth(record), np.int64), record
            mask_from_csv = self.masks[record.split('-')[1]]
            mask_from_file = self.convert_to_run_length_encoding(self.mask(record))
            if isinstance(mask_from_csv, str):
                assert mask_from_csv == mask_from_file, record
            else:
                assert np.isnan(mask_from_file), record

        for record in self.test:
            assert isfile(self.path(record, 'image')), record

        assert len(self.test) == 18000
        assert len(self.train) == 4000


    def image(self, record, augmentor=lambda x: x):
        img = imread(self.path(record, 'image'))
        img = augmentor(img)
        return img


    def mask(self, record, augmentor=lambda x: x):
        mask = imread(self.path(record, 'mask'))
        mask = augmentor(mask)
        return mask


    def depth(self, record):
        mode, image_id = record.split('-', 1)
        #assert mode == 'train'
        return self.depths[image_id]


    def salt(self, record):
        mode, image_id = record.split('-')
        assert mode == 'train'
        return isinstance(self.masks[image_id], str)


    def coverage_split(self, record):
        mask = self.mask(record)
        salt_coverage = np.count_nonzero(mask) / mask.size
        split_size = 100 / self.coverage_splits
        return salt_coverage // split_size


    def path(self, record, kind):
        mode, image_id = record.split('-', 1)
        return '%s/%s/%ss/%s.png' % (self.root, mode, kind, image_id)


    def convert_to_run_length_encoding(self, img):
        flat_img = img.transpose().flatten()
        flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)
        flat_img = np.insert(flat_img, [0, len(flat_img)], [0, 0])

        starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
        ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))

        if starts.any():
            starts_ix = np.where(starts)[0] + 1
            ends_ix = np.where(ends)[0] + 1
            lengths = ends_ix - starts_ix

            encoding = ''
            for idx in range(len(starts_ix)):
                encoding += '%d %d ' % (starts_ix[idx], lengths[idx])
            return encoding.strip()
        else:
            return np.float('nan')



if __name__ == '__main__':
    import sys
    Dataset(sys.argv[1]).check_integrity()
