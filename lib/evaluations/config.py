# --------------------------------------------------------
# Adapted from Ross Girshick's Fast/er R-CNN code
# --------------------------------------------------------

import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Detector options [legacy support]
# These options are only used for the Basic evaluation method, if not specified in the eval scripts directly.
#

__C.TEST = edict()

# Number classes considered during testing
__C.TEST.NUM_CLASSES = 240

# Min score for any class
# (if not any score larger than thresh, suppress box)
__C.TEST.SCORE_MIN_THRESH = 0.05  # 0.01

# Score threshold for ROI to be considered background
# (if bg score in (THRESH, 1], suppress box)
__C.TEST.SCORE_BG_THRESH = 0.7

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Test using bounding-box regressors (only works if a network trained for bbox_reg is evaluated)
__C.TEST.BBOX_REG = True

# Shift applied to the five different crops during oversampling
__C.TEST.SHIFT = 24

# Min overlap with ground truth box for positive detection (if IoU < this threshold, detection is a false positive)
__C.TEST.TP_MIN_OVERLAP = 0.5  # 0.4


# Data directory
__C.DATA_DIR = '/home/tobias/Datasets/cuneiform/'

# tablet directories
__C.DATA_TEST_DIR = __C.DATA_DIR + 'test_images/'



# FUNCTIONS for loading cfg
#


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if k not in b:  # not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d  # d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d  # d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value