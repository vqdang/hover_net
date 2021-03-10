import re
import subprocess
import warnings
from collections import OrderedDict

import cv2
import glymur
import numpy as np
import openslide
import tifffile
from skimage import color, img_as_ubyte


class FileHandler(object):
    def __init__(self):
        """
        The handler is responsible for storing the processed data, parsing
        the metadata from original file, and reading it from storage. 
        """
        self.metadata = {
            ('available_mag', None),
            ('base_mag'     , None),
            ('vendor'       , None),
            ('base_mpp'     , None),
            ('base_shape'   , None),
        }

        pass

    def __load_metadata(self):
        pass

    def read_region(self):
        pass
    
    def get_dimensions(self, read_mag=None, read_mpp=None):
        """
        Will be in X, Y
        """
        if read_mpp is not None:
            read_scale = (self.metadata['base_mpp'] / read_mpp)[0]
            read_mag = read_scale * self.metadata['base_mag']
        scale = read_mag / self.metadata['base_mag']
        # may off some pixels wrt existing mag
        return (self.metadata['base_shape'] * scale).astype(np.int32)

    def prepare_reading(self, read_mag=None, read_mpp=None):
        """
        Only use either of these parameter, prioritize `read_mpp`

        `read_mpp` is in X, Y format
        """
        hires_lv, scale_from_hires_lv, scale_from_lv0 = self._get_read_info(read_mag=read_mag, read_mpp=read_mpp)  

        self.read_lv = hires_lv
        self.scale_from_hires_lv = scale_from_hires_lv
        self.scale_from_lv0 = scale_from_lv0
        return

    def _get_read_info(self, read_mag=None, read_mpp=None):
        if read_mpp is not None:
            assert read_mpp[0] == read_mpp[1], 'Not supported uneven `read_mpp`'
            read_scale = (self.metadata['base_mpp'] / read_mpp)[0]
            read_mag = read_scale * self.metadata['base_mag']

        hires_mag = read_mag
        scale_from_lv0 = read_mag / self.metadata['base_mag']
        scale_from_hires_lv = scale_from_lv0
        if read_mag not in self.metadata['available_mag']:
            if read_mag > self.metadata['base_mag']:
                scale_from_hires_lv = scale_from_lv0
                hires_mag = self.metadata['base_mag']
            else:          
                mag_list = np.array(self.metadata['available_mag'])
                mag_list = np.sort(mag_list)[::-1]
                hires_mag = mag_list - read_mag
                # only use higher mag as base for loading
                hires_mag = hires_mag[hires_mag > 0]
                # use the immediate higher to save compuration
                hires_mag = mag_list[np.argmin(hires_mag)]
                scale_from_hires_lv = read_mag / hires_mag

        hires_lv = self.metadata['available_mag'].index(hires_mag) 
        return hires_lv, scale_from_hires_lv, scale_from_lv0

class OpenSlideHandler(FileHandler):
    """
    Class for handling OpenSlide supported whole-slide images
    """
    def __init__(self, file_path):
        """
        file_path (string): path to single whole-slide image
        """
        super().__init__()
        self.file_ptr = openslide.OpenSlide(file_path) # load OpenSlide object
        self.metadata = self.__load_metadata()

        # only used for cases where the read magnification is different from
        self.image_ptr = None # the existing modes of the read file
        self.read_level = None

    def __load_metadata(self):
        metadata = {}

        wsi_properties = self.file_ptr.properties
        mpp = [float(wsi_properties[openslide.PROPERTY_NAME_MPP_X]),
               float(wsi_properties[openslide.PROPERTY_NAME_MPP_Y])]
        mpp = np.array(mpp)

        try :
            level_0_magnification = wsi_properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            level_0_magnification = float(level_0_magnification)
            downsample_level = self.file_ptr.level_downsamples
            magnification_level = [level_0_magnification / lv for lv in downsample_level]
        except:
            if mpp[0] > 0.1 and mpp[1] < 0.4:
                level_0_magnification = 40.0
            if mpp[0] >= 0.4 and mpp[1] < 0.6:
                level_0_magnification = 20.0
            downsample_level = self.file_ptr.level_downsamples
            warnings.warn('Could not detect magnification, guess from `mpp`.')
        magnification_level = [level_0_magnification / lv for lv in downsample_level]

        metadata = [
            ('available_mag', magnification_level), # highest to lowest mag
            ('base_mag'     , magnification_level[0]),
            ('vendor'       , wsi_properties[openslide.PROPERTY_NAME_VENDOR]),
            ('base_mpp'     , mpp),
            ('base_shape'   , np.array(self.file_ptr.dimensions)),
        ]
        return OrderedDict(metadata)
    
    def read_region(self, coords, size):
        """
        Must call `prepare_image` before hand

        Args:
            coords (tuple): top left coordinates of image region at requested mag/mpp in `prepare_reading`
            read_level_size (tuple): dimensions of image region at requested mag/mpp in `prepare_reading` (dims_x, dims_y)
        """

        # convert coord from read lv to lv zero
        coord_lv0 = [0, 0]
        coord_lv0[0] = int(coords[0] / self.scale_from_lv0)
        coord_lv0[1] = int(coords[1] / self.scale_from_lv0)

        size_at_read_lv = (size / self.scale_from_hires_lv).astype(np.int32)
        region = self.file_ptr.read_region(coord_lv0, self.read_lv, size_at_read_lv)

        region = np.array(region)[...,:3]
        if self.scale_from_hires_lv is not None:
            interp = cv2.INTER_LINEAR
            region = cv2.resize(region, tuple(size), interpolation=interp)            
        return region

    def get_full_img(self, read_mag=None, read_mpp=None):

        read_lv, scale_from_hires_lv, scale_from_lv0 = self._get_read_info(
                                        read_mag=read_mag, 
                                        read_mpp=read_mpp)        

        read_size = self.file_ptr.level_dimensions[read_lv]

        wsi_img = self.file_ptr.read_region((0, 0), read_lv, read_size)
        wsi_img = np.array(wsi_img)[...,:3] # remove alpha channel
        if scale_from_hires_lv is not None:
            # now rescale then return
            interp = cv2.INTER_LINEAR
            wsi_img = cv2.resize(wsi_img, (0, 0), 
                            fx=scale_from_hires_lv, 
                            fy=scale_from_hires_lv,
                            interpolation=interp)
        return wsi_img         

def get_file_handler(path, backend):
    if backend in [
            '.svs', '.tif', 
            '.vms', '.vmu', '.ndpi',
            '.scn', '.mrxs', '.tiff',
            '.svslide',
            '.bif',
            ]:
        return OpenSlideHandler(path)
    else:
        assert False, "Unknown WSI format `%s`" % backend

