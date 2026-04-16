import numpy as np


def augment(input_image):
    '''
    Augments an image using rotation and flipping. Returns a list of 7 images:
        0 - Original
        1 - Rotated 90° clockwise
        2 - Rotated 180°
        3 - Rotated 270° clockwise
        4 - Flipped vertically
        5 - Flipped horizontally
        6 - Flipped vertically & horizontally

    input_image may be single-channel (H, W) or multi-channel (H, W, C).
    '''
    return [
        input_image,
        np.rot90(input_image, k=3),    # 90° clockwise
        np.rot90(input_image, k=2),    # 180°
        np.rot90(input_image, k=1),    # 270° clockwise
        np.flip(input_image, axis=0),  # vertical flip
        np.flip(input_image, axis=1),  # horizontal flip
        np.flip(input_image, axis=(0, 1)),  # both axes
    ]
