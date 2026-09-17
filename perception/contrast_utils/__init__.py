"""PCD-Mask perception utilities that produce inpainted contrast images."""

from .contrast_image_generator import ContrastImageGenerator

def get_contrast_image_generator(config):
    return ContrastImageGenerator(**config)
