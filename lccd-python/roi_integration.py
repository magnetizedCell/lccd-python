import itertools

import numpy as np

def filter_roi_by_area(roi, min_area, max_area):
    n_rois = roi.shape[1]
    indices = []
    for i in range(n_rois):
        area = np.sum(roi[:, i])
        if min_area <= area and area <= max_area:
            indices.append(i)
    return np.stack([roi[:, i] for i in indices], 1)


class RoiIntegration():
    def __init__(self, overlap_threshold, min_area, max_area, **kwargs):
        self.overlap_threshold = overlap_threshold
        self.min_area = min_area
        self.max_area = max_area

    def apply(self, roi_a, roi_b):
        # requires test
        if roi_b is None:
            return roi_a
        dtype = roi_a.dtype
        while 1:
            change_occured = False
            sim = np.dot(roi_a.T, roi_b)
            # sim[i][j] = area of intersection of ith ROI in roi_a and jth ROI in roi_b
            # TODO: Try sparse matrix for sim
            # Are roi_a and roi_b sparse themselves?

            for region_in_roi_a in range(roi_a.shape[1]):
                region_in_roi_b = np.argmax(sim[region_in_roi_a]) # most overlapping region in roi_b
                area_overlap = sim[region_in_roi_a][region_in_roi_b]
                if area_overlap == 0:
                    continue
                area_a = np.sum(roi_a[:, region_in_roi_a])
                area_b = np.sum(roi_b[:, region_in_roi_b])

                if area_overlap / area_a > self.overlap_threshold or\
                        area_overlap / area_b > self.overlap_threshold:
                    change_occured = True
                    if area_a > area_b:
                        roi_b = np.delete(roi_b, region_in_roi_b, 1)
                        # column deletion from large array is slow.
                    else:
                        roi_a = np.delete(roi_a, region_in_roi_a, 1)
                    break
                elif area_overlap / area_a <= self.overlap_threshold and\
                        area_overlap / area_b <= self.overlap_threshold:
                    change_occured = True
                    region_overlap = (roi_a[:, region_in_roi_a] * roi_b[:, region_in_roi_b]) == 1
                    region_non_overlap = (1 - region_overlap).astype(dtype)
                    roi_a[:, region_in_roi_a] *= region_non_overlap
                    roi_b[:, region_in_roi_b] *= region_non_overlap
                    break
                else:
                    continue

            if not change_occured:
                break

        roi = np.concatenate([roi_a, roi_b], 1)
        roi = filter_roi_by_area(roi, self.min_area, self.max_area)
        return roi
