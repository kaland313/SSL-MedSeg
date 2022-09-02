import numpy as np
from tabulate import tabulate

def pad_to_next_multiple_of_32(img):
    """
    Pad each dim to multiples of 2^5 = 32
    Assuming the encoder is a ResNet with 5 blocks, it max pools images 5 times.
    To avoid runtime errors images are padded to the the next multiple of 32. Separately for each dimension

    Args:
        img ([np.ndarray]): Image (1 channel).
    """
    pad = []
    for dim in img.shape[1:]:
        if dim % 32 == 0:
            pad.append(0)
        else:
            pad.append(32 - dim % 32)
    return np.pad(img, ((0,0),(0, pad[0]),(0, pad[1])))
    