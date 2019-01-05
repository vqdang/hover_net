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

from tensorpack.dataflow.imgaug import ImageAugmentor
from tensorpack.utils.utils import get_rng

from misc.utils import cropping_center, bounding_box

####
class GenInstance(ImageAugmentor):
    def __init__(self, crop_shape=None):
        super(GenInstance, self).__init__()
        self.crop_shape = crop_shape
    
    def reset_state(self):
        self.rng = get_rng(self)

    def _fix_mirror_padding(self, ann):
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
####
class GenInstanceXY(GenInstance):   
    """
        Input annotation must be of original shape.
        The map is calculated only for instances within the crop portion
        but basing on the original shape in original image
    """

    def _augment(self, img, _):
        img = np.copy(img)
        orig_ann = img[...,0] # instance ID map
        fixed_ann = self._fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
        crop_ann = cropping_center(fixed_ann, self.crop_shape)
        # removing rear pixels that are too small for balancing com
        if crop_ann.sum() > 0: # to avoide the 1 label warning
            crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

        x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(crop_ann))
        inst_list.remove(0) # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(fixed_ann == inst_id, np.uint8)
            inst_box = bounding_box(inst_map)

            # expand the box by 2px
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

        orig_ann = orig_ann.astype('float32')
        img = np.dstack([orig_ann, x_map, y_map])

        return img

####
import matplotlib.pyplot as plt
class GenInstanceDistance(GenInstance):   
    """
        Input annotation must be of original shape.
        The map is calculated only for instances within the crop portion
        but basing on the original shape in original image
    """
    def _augment(self, img, _):
        img = np.copy(img)
        orig_ann = img[...,0] # instance ID map
        fixed_ann = self._fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
        crop_ann = cropping_center(fixed_ann, self.crop_shape)

        orig_dst = np.zeros(orig_ann.shape, dtype=np.float32)  

        inst_list = list(np.unique(crop_ann))
        inst_list.remove(0) # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(fixed_ann == inst_id, np.uint8)
            inst_box = bounding_box(inst_map)

            # expand the box by 2px
            inst_box[0] -= 2
            inst_box[2] -= 2
            inst_box[1] += 2
            inst_box[3] += 2

            inst_map = inst_map[inst_box[0]:inst_box[1],
                                inst_box[2]:inst_box[3]]

            if inst_map.shape[0] < 2 or \
                inst_map.shape[1] < 2:
                continue

            # chessboard distance map generation
            # normalize distance to 0-1
            inst_dst = distance_transform_cdt(inst_map)
            inst_dst = inst_dst.astype('float32')
            inst_dst = (inst_dst / np.amax(inst_dst)) 

            ####
            dst_map_box = orig_dst[inst_box[0]:inst_box[1],
                                   inst_box[2]:inst_box[3]]
            dst_map_box[inst_map > 0] = inst_dst[inst_map > 0]

        #
        orig_ann = orig_ann.astype('float32')
        img = np.dstack([orig_ann, orig_dst])
        
        return img

####
class GaussianBlur(ImageAugmentor):
    """ Gaussian blur the image with random window size"""
    def __init__(self, max_size=3):
        """
        Args:
            max_size (int): max possible Gaussian window size would be 2 * max_size + 1
        """
        super(GaussianBlur, self).__init__()
        self.max_size = max_size

    def _get_augment_params(self, img):
        sx, sy = self.rng.randint(1, self.max_size, size=(2,))
        sx = sx * 2 + 1
        sy = sy * 2 + 1
        return sx, sy

    def _augment(self, img, s):
        return np.reshape(cv2.GaussianBlur(img, s, sigmaX=0, sigmaY=0,
                                           borderType=cv2.BORDER_REPLICATE), img.shape)

####
class BinarizeLabel(ImageAugmentor):
    """ Convert labels to binary maps"""
    def __init__(self):
        super(BinarizeLabel, self).__init__()

    def _get_augment_params(self, img):
        return None

    def _augment(self, img, s):
        img = np.copy(img)
        arr = img[...,0]
        arr[arr > 0] = 1
        return img

####
class MedianBlur(ImageAugmentor):
    """ Median blur the image with random window size"""
    def __init__(self, max_size=3):
        """
        Args:
            max_size (int): max possible window size 
                            would be 2 * max_size + 1
        """
        super(MedianBlur, self).__init__()
        self.max_size = max_size

    def _get_augment_params(self, img):
        s = self.rng.randint(1, self.max_size)
        s = s * 2 + 1
        return s

    def _augment(self, img, ksize):
        return cv2.medianBlur(img, ksize)

#### TODO: add test run here
# if __name__ == '__main__':
    
#     # for debugging
#     trans = GenInstanceXY()
#     trans.reset_state()

#     img = cv2.imread('sample.png')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     out = trans.test_run(img)
#     out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
#     cv2.imwrite('out.png', out)