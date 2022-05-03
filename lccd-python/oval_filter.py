import numpy as np
import skimage


def oval_filter(label_mat):
    rois = []
    regionprops = skimage.measure.regionprops(label_mat)
    for regionprop in regionprops:
        area_ratio = regionprop.axis_major_length * regionprop.axis_minor_length* np.pi / 4 / regionprop.area
        eccen = regionprop.eccentricity
        if area_ratio < 1.8 and eccen < 0.99: # TODO Make configurable
            roi = (label_mat == regionprop.label).astype(np.uint8)
            roi = roi.reshape(-1, 1)
            rois.append(roi)
    rois = np.concatenate(rois, 1)
    return rois
