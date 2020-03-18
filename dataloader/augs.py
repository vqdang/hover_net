import math

import cv2
import matplotlib.cm as cm
import numpy as np

from scipy import ndimage
from scipy.ndimage import measurements
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import affine_transform, map_coordinates
from scipy.ndimage.morphology import (distance_transform_cdt,
                                      distance_transform_edt)
from skimage import morphology as morph

from misc.utils import cropping_center, bounding_box

def fix_mirror_padding(ann):
    """
    Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.)
    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0) # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann

def gen_instance_hv_map(ann, crop_shape):
    """
        Input annotation must be of original shape.
        
        The map is calculated only for instances within the crop portion
        but based on the original shape in original image.
    
        Perform following operation:
        Obtain the horizontal and vertical distance maps for each
        nuclear instance.
    """

    orig_ann = ann.copy() # instance ID map
    fixed_ann = fix_mirror_padding(orig_ann)
    # re-cropping with fixed instance id map
    crop_ann = cropping_center(fixed_ann, crop_shape)
    # TODO: deal with 1 label warning
    crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0) # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inst_box = bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0]:inst_box[1],
                            inst_box[2]:inst_box[3]]

        if inst_map.shape[0] < 2 or \
            inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))
        
        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1]+1)
        inst_y_range = np.arange(1, inst_map.shape[0]+1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]
        
        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype('float32')
        inst_y = inst_y.astype('float32')

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= (-np.amin(inst_x[inst_x < 0]))
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= (-np.amin(inst_y[inst_y < 0]))
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= (np.amax(inst_x[inst_x > 0]))
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= (np.amax(inst_y[inst_y > 0]))

        ####
        x_map_box = x_map[inst_box[0]:inst_box[1],
                            inst_box[2]:inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0]:inst_box[1],
                            inst_box[2]:inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map])
    return hv_map