import random
import torch
import numpy as np


global RANDOM_SEED
RANDOM_SEED = None


def set_random_seed(seed=None):
    global RANDOM_SEED
    changed = False
    if RANDOM_SEED is None:
        RANDOM_SEED = random.randint(1, 10000)
        changed = True
    if seed is not None:
        RANDOM_SEED = seed
        changed = True
    if changed:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        print('Random seed changed: %i' % RANDOM_SEED)
    return RANDOM_SEED


def base_model_name(arch_name, random_seed=None):
    seed = random_seed if random_seed else set_random_seed()
    return 'seed-%i_%s' % (seed, arch_name)
