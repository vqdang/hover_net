
import cv2
import numpy as np
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed

####
def proc_np_dist(pred):
    """
    Process Nuclei Prediction with Distance Map

    Args:
        pred: prediction output, assuming 
                channel 0 contain probability map of nuclei
                channel 1 containing the regressed distance map
    """
    blb_raw = pred[...,0]
    dst_raw = pred[...,1]

    blb = np.copy(blb_raw)
    blb[blb >  0.5] = 1
    blb[blb <= 0.5] = 0
    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1   

    dst_raw[dst_raw < 0] = 0
    dst = np.copy(dst_raw)
    dst = dst * blb
    dst[dst  > 0.5] = 1
    dst[dst <= 0.5] = 0

    marker = dst.copy()
    marker = binary_fill_holes(marker) 
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
    proced_pred = watershed(-dst_raw, marker, mask=blb)    
    return proced_pred

####
def proc_np_hv(pred, marker_mode=2, energy_mode=2, rgb=None):
    """
    Process Nuclei Prediction with XY Coordinate Map

    Args:
        pred: prediction output, assuming 
                channel 0 contain probability map of nuclei
                channel 1 containing the regressed X-map
                channel 2 containing the regressed Y-map
    """
    assert marker_mode == 2 or marker_mode == 1, 'Only support 1 or 2'
    assert energy_mode == 2 or energy_mode == 1, 'Only support 1 or 2'

    blb_raw = pred[...,0]
    h_dir_raw = pred[...,1]
    v_dir_raw = pred[...,2]

    ##### Processing 
    blb = np.copy(blb_raw)
    blb[blb >= 0.5] = 1
    blb[blb <  0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1 # back ground is 0 already
    #####

    if energy_mode == 2 or marker_mode == 2:
        h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

        sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        if energy_mode == 2:
            dist = (1.0 - overall) * blb
            ## nuclei values form mountains so inverse to get basins
            dist = -cv2.GaussianBlur(dist,(3, 3),0)

        if marker_mode == 2:
            overall[overall >= 0.4] = 1
            overall[overall <  0.4] = 0
            
            marker = blb - overall
            marker[marker < 0] = 0
            marker = binary_fill_holes(marker).astype('uint8')
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
            marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
            marker = measurements.label(marker)[0]
            marker = remove_small_objects(marker, min_size=10)

    if energy_mode == 1:
        dist = h_dir_raw * h_dir_raw + v_dir_raw * v_dir_raw
        dist[blb == 0] = np.amax(dist)
        # nuclei values are already basins
        dist = filters.maximum_filter(dist, 7)
        dist = cv2.GaussianBlur(dist, (3, 3),0)

    if marker_mode == 1:
        h_marker = np.copy(h_dir_raw)
        v_marker = np.copy(v_dir_raw)
        h_marker = np.logical_and(h_marker <  0.075, h_marker > -0.075)
        v_marker = np.logical_and(v_marker <  0.075, v_marker > -0.075)
        marker = np.logical_and(h_marker > 0, v_marker > 0) * blb
        marker = binary_dilation(marker, iterations=2)
        marker = binary_fill_holes(marker) 
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=10)
    
    proced_pred = watershed(dist, marker, mask=blb)

    return proced_pred
