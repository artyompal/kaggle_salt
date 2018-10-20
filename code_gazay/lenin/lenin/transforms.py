import numpy as np

def hwc_to_chw(image):
        return np.einsum('hwc->chw', image) # change to pytorch format
