import albumentations
import numpy as np
import copy, torch

class Augmentor:
    def __init__(self, augmentations):
        self.augmentations = [self.augmentation(it) for it in augmentations]

    def transformations(self):
        transformations = []
        for augmentation in self.augmentations:
            if np.random.random() < augmentation.p:
                fixed_params = augmentation.get_params()
                transformation = copy.copy(augmentation)
                transformation.p = 1
                transformation.get_params = lambda: fixed_params
                transformations.append(transformation)
        return Transformations(transformations)

    # we will need our primitives for coordinating (one_of, one_or_other,...)
    def __call__(self, image):
        return self.transformations()(image)

    def augmentation(self, definition):
        return getattr(albumentations, definition.pop('type'))(**definition)


class Transformations:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, image):
        if torch.is_tensor(image):
            image = image.numpy()
        for transformation in self.transformations:
            image = transformation(image=image)['image']
        return image


if __name__ == '__main__':
    image = np.ones((300, 300, 3), dtype=np.uint8)
    Augmentor({ 'type': 'GaussNoise', 'p': 1 }, { 'type': 'Flip' })(image)
