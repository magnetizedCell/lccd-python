import scipy.signal
import numpy as np

def matlab_conv2(x, y, mode='same'):
    if mode == 'same':
        # https://stackoverflow.com/a/38355889
        return np.rot90(scipy.signal.convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode='same'), 2)
    if mode == 'valid':
        # https://stackoverflow.com/questions/43270274/equivalent-of-matlab-filter2filter-image-valid-in-python
        # Still there is some difference
        return scipy.signal.convolve2d(x, np.rot90(y, 2), mode='valid')
