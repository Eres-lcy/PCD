"""PCD-Fast perception utilities that produce target segmentation masks."""

from .segmented_image_generator import SegmentedImageGenerator

def get_segmented_image_generator(config):
    return SegmentedImageGenerator(**config)
