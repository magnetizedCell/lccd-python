import numpy as np
import scipy
import scipy.signal
import skimage
import skimage.measure
import skimage.filters
import skimage.exposure

from .oval_filter import oval_filter
from .utils import matlab_conv2

try:
    import cupy as cp
    import cupyx.scipy.signal
except ImportError:
    pass

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    From
    https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def im2bw(I, level):
    result = np.zeros_like(I)
    result[I > level] = 1
    return result


def filter_closed_regions_by_area(mat, min_area, max_area):
    mat = np.copy(mat)
    num_regions = np.max(mat)
    for i in range(1, 1+num_regions):
        area = np.sum(mat==i)
        if not(min_area < area and area <= max_area):
            mat[mat==i] = 0
    return mat


class BlobDetector():
    def __init__(self, filtersize1, filtersize2, sigma, fsize, min_area, max_area, gpu=False,
                matlab_conv=False, debug=False, **kwargs):
        self.min_area = min_area
        self.max_area = max_area
        self.mavefil1 = np.ones((1, filtersize1)) / filtersize1
        self.mavefil2 = np.ones((1, filtersize2)) / filtersize2
        self.sp_fil = matlab_style_gauss2D(shape=(fsize, fsize), sigma=sigma)\
                        - np.ones((fsize, fsize)) / (fsize**2) # average filter
        self.conv2 = matlab_conv2 if matlab_conv else scipy.signal.convolve2d
        self.debug = debug
        self.gpu = gpu
        if gpu:
            self.mavefil1 = cp.asarray(self.mavefil1)
            self.mavefil2 = cp.asarray(self.mavefil2)
            self.sp_fil = cp.asarray(self.sp_fil)
            self.conv2 = cupyx.scipy.signal.convolve2d
            self.xp = cp
        else:
            self.xp = np

    def preprocess(self, tv, X, Y):
        if self.gpu:
            tv = cp.asarray(tv)
        smooth_tv = self.xp.maximum(
            self.conv2(tv, self.mavefil2, mode='same')\
                - self.conv2(tv, self.mavefil1, mode='same'),
            0
        )
        max_lumi_map = self.xp.max(smooth_tv, axis=1).reshape(X, Y)
        lumi_map = self.xp.maximum(
            self.conv2(max_lumi_map, self.sp_fil, mode='valid'),
            # Here, the article says the mode is 'valid' (Algorithm2, line 10).
            # But in order that ROI have the dimesion (0, 1)^{XY x N_r} (line 21),
            # mode should be 'same'.
            0)
        if self.gpu:
            lumi_map = cp.asnumpy(lumi_map)
        if not self.debug:
            return lumi_map
        if self.gpu:
            smooth_tv = cp.asnumpy(smooth_tv)
            max_lumi_map = cp.asnumpy(max_lumi_map)
        return lumi_map, smooth_tv, max_lumi_map

    def apply(self, tv):
        X, Y, T = tv.shape
        tv = tv.reshape(-1, T)
        if self.debug:
            lumi_map, smooth_tv, max_lumi_map = self.preprocess(tv, X, Y)
        else:
            lumi_map = self.preprocess(tv, X, Y)
        cont_enhanced_map = skimage.exposure.equalize_adapthist(lumi_map)
        threshold = skimage.filters.threshold_otsu(cont_enhanced_map)
        bwm = im2bw(cont_enhanced_map, threshold)
        label_mat = skimage.measure.label(bwm, connectivity=2)
        label_mat2 = filter_closed_regions_by_area(
            label_mat, min_area=self.min_area, max_area=self.max_area)
        ROI = oval_filter(label_mat2)
        if not self.debug:
            return ROI
        return {
            'smooth_tv': smooth_tv,
            'max_lumi_map': max_lumi_map,
            'lumi_map': lumi_map,
            'cont_enhanced_map': cont_enhanced_map,
            'threshold': threshold,
            'bwm': bwm,
            'label_mat': label_mat,
            'label_mat2': label_mat2,
            'ROI': ROI
            }
