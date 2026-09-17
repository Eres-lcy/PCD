import numpy as np


def index_all(l, e):
    # find all indexs of e in l
    return [i for i, x in enumerate(l) if x == e]


def mask_to_bbox(mask):
    y, x = np.where(mask)
    return np.array([x.min(), y.min(), x.max(), y.max()])

