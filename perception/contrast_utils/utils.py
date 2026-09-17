import cv2
import numpy as np


def dilate_mask(mask, kernel_size):
    if mask is None:
        return None
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def index_all(l, e):
    # find all indexs of e in l
    return [i for i, x in enumerate(l) if x == e]


def mask_to_bbox(mask):
    y, x = np.where(mask)
    return np.array([x.min(), y.min(), x.max(), y.max()])

