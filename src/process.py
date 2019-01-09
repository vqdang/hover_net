
import glob
import os

import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from skimage.morphology import remove_small_objects, watershed

from config import Config
from misc.viz_utils import visualize_instances

def proc_np_dst(pred):
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
    blb = remove_small_objects(blb, min_size=30)
    blb[blb > 0] = 1   

    dst_raw[dst_raw < 0] = 0
    dst = np.copy(dst_raw)
    dst = dst * blb
    dst[dst  > 0.5] = 1
    dst[dst <= 0.5] = 0

    marker = dst.copy()
    marker = binary_fill_holes(marker) 
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=30)
    proced_pred = watershed(-dst_raw, marker, mask=blb)    
    return proced_pred

def proc_np_xy(pred, marker_mode=2, energy_mode=2):
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
    x_dir_raw = pred[...,1]
    y_dir_raw = pred[...,2]

    ##### Processing 
    blb = np.copy(blb_raw)
    blb[blb >= 0.5] = 1
    blb[blb <  0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=30)
    blb[blb > 0] = 1 # back ground is 0 already
    #####

    if energy_mode == 2 or marker_mode == 2:
        x_dir = cv2.normalize(x_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        y_dir = cv2.normalize(y_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        sobelx = cv2.Sobel(x_dir, cv2.CV_64F, 1, 0, ksize=21)
        sobely = cv2.Sobel(y_dir, cv2.CV_64F, 0, 1, ksize=21)

        sobelx = 1 - (cv2.normalize(sobelx, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        sobely = 1 - (cv2.normalize(sobely, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

        overall = np.maximum(sobelx, sobely)
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
            marker = remove_small_objects(marker, min_size=30)

    if energy_mode == 1:
        dist = x_dir_raw * x_dir_raw + y_dir_raw * y_dir_raw
        dist[blb == 0] = np.amax(dist)
        # nuclei values are already basins
        dist = filters.maximum_filter(dist, 7)
        dist = cv2.GaussianBlur(dist, (3, 3),0)

    if marker_mode == 1:
        x_marker = np.copy(x_dir_raw)
        y_marker = np.copy(y_dir_raw)
        x_marker = np.logical_and(x_marker <  0.075, x_marker > -0.075)
        y_marker = np.logical_and(y_marker <  0.075, y_marker > -0.075)
        marker = np.logical_and(x_marker > 0, y_marker > 0) * blb
        marker = binary_dilation(marker, iterations=2)
        marker = binary_fill_holes(marker) 
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=10)
    
    # print(blb.shape, dist.shape, marker.shape)
    proced_pred = watershed(dist, marker, mask=blb)
    return proced_pred

###################

## WARNING: check the prediction channels, wrong ordering will break the code !
## It should match augs.py

cfg = Config()

proc_mode = 'np+xy'

# 1 - threshold, 2 - sobel based
energy_mode = 2 
marker_mode = 2 

for norm_target in cfg.inf_norm_codes:
    imgs_dir = '%s/XXXX/' % (cfg.inf_norm_root_dir)            
    # pred_dir = '%s/%s/' % (cfg.inf_output_dir, norm_target)
    # proc_dir = '%s/%s_proc/' % (cfg.inf_output_dir, norm_target)

    pred_dir = 'output/v6.0.3.0/Kumar/xy_temp/XXXX/'
    proc_dir = 'output/v6.0.3.0/Kumar/xy_temp/XXXX_proc/'

    # TODO: cache list to check later norm dir has same number of files
    file_list = glob.glob('%s/*.mat' % (pred_dir))
    file_list.sort() # ensure same order

    if not os.path.isdir(proc_dir):
        os.makedirs(proc_dir)

    for filename in file_list:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]
        print(basename, norm_target, end=' ')

        ##
        img = cv2.imread(imgs_dir + basename + cfg.inf_imgs_ext)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        pred = sio.loadmat('%s/%s.mat' % (pred_dir, basename))
        pred = np.squeeze(pred['result'])
        if proc_mode == 'np+xy':
            pred = proc_np_xy(pred, 
                            marker_mode=marker_mode,
                            energy_mode=energy_mode)
        if proc_mode == 'np+dst':
            pred = proc_np_dst(pred)

        ##
        overlaid_output = visualize_instances(pred, img)
        overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
        cv2.imwrite('%s/%s.png' % (proc_dir, basename), overlaid_output)

        sio.savemat('%s/%s_predicted_map.mat' % (proc_dir, basename), 
                                    {'predicted_map': pred})

        ##
        print('FINISH')