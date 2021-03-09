from collections import OrderedDict
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage import color
import re
import subprocess

import openslide


class FileHandler(object):
    def __init__(self):
        """The handler is responsible for storing the processed data, parsing
        the metadata from original file, and reading it from storage. 
        """
        self.metadata = {
            ("available_mag", None),
            ("base_mag", None),
            ("vendor", None),
            ("mpp  ", None),
            ("base_shape", None),
        }
        pass

    def __load_metadata(self):
        raise NotImplementedError

    def get_full_img(self, read_mag=None, read_mpp=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.

        `read_mpp` is in X, Y format
        """
        raise NotImplementedError

    def read_region(self, coords, size):
        """Must call `prepare_reading` before hand.

        Args:
            coords (tuple): (dims_x, dims_y), 
                          top left coordinates of image region at selected 
                          `read_mag` or `read_mpp` from `prepare_reading` 
            size (tuple): (dims_x, dims_y)
                          width and height of image region at selected 
                          `read_mag` or `read_mpp` from `prepare_reading`       

        """
        raise NotImplementedError

    def get_dimensions(self, read_mag=None, read_mpp=None):
        """Will be in X, Y."""
        if read_mpp is not None:
            read_scale = (self.metadata["base_mpp"] / read_mpp)[0]
            read_mag = read_scale * self.metadata["base_mag"]
        scale = read_mag / self.metadata["base_mag"]
        # may off some pixels wrt existing mag
        return (self.metadata["base_shape"] * scale).astype(np.int32)

    def prepare_reading(self, read_mag=None, read_mpp=None, cache_path=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.

        `read_mpp` is in X, Y format.
        """
        read_lv, scale_factor = self._get_read_info(
            read_mag=read_mag, read_mpp=read_mpp
        )

        if scale_factor is None:
            self.image_ptr = None
            self.read_lv = read_lv
        else:
            np.save(cache_path, self.get_full_img(read_mag=read_mag))
            self.image_ptr = np.load(cache_path, mmap_mode="r")
        return

    def _get_read_info(self, read_mag=None, read_mpp=None):
        if read_mpp is not None:
            assert read_mpp[0] == read_mpp[1], "Not supported uneven `read_mpp`"
            read_scale = (self.metadata["base_mpp"] / read_mpp)[0]
            read_mag = read_scale * self.metadata["base_mag"]

        hires_mag = read_mag
        scale_factor = None
        if read_mag not in self.metadata["available_mag"]:
            if read_mag > self.metadata["base_mag"]:
                scale_factor = read_mag / self.metadata["base_mag"]
                hires_mag = self.metadata["base_mag"]
            else:
                mag_list = np.array(self.metadata["available_mag"])
                mag_list = np.sort(mag_list)[::-1]
                hires_mag = mag_list - read_mag
                # only use higher mag as base for loading
                hires_mag = hires_mag[hires_mag > 0]
                # use the immediate higher to save compuration
                hires_mag = mag_list[np.argmin(hires_mag)]
                scale_factor = read_mag / hires_mag

        hires_lv = self.metadata["available_mag"].index(hires_mag)
        return hires_lv, scale_factor


class OpenSlideHandler(FileHandler):
    """Class for handling OpenSlide supported whole-slide images."""

    def __init__(self, file_path):
        """file_path (string): path to single whole-slide image."""
        super().__init__()
        self.file_ptr = openslide.OpenSlide(file_path)  # load OpenSlide object
        self.metadata = self.__load_metadata()

        # only used for cases where the read magnification is different from
        self.image_ptr = None  # the existing modes of the read file
        self.read_level = None

    def __load_metadata(self):
        metadata = {}

        wsi_properties = self.file_ptr.properties
        level_0_magnification = wsi_properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        level_0_magnification = float(level_0_magnification)

        downsample_level = self.file_ptr.level_downsamples
        magnification_level = [level_0_magnification / lv for lv in downsample_level]

        mpp = [
            wsi_properties[openslide.PROPERTY_NAME_MPP_X],
            wsi_properties[openslide.PROPERTY_NAME_MPP_Y],
        ]
        mpp = np.array(mpp)

        metadata = [
            ("available_mag", magnification_level),  # highest to lowest mag
            ("base_mag", magnification_level[0]),
            ("vendor", wsi_properties[openslide.PROPERTY_NAME_VENDOR]),
            ("mpp  ", mpp),
            ("base_shape", np.array(self.file_ptr.dimensions)),
        ]
        return OrderedDict(metadata)

    def read_region(self, coords, size):
        """Must call `prepare_reading` before hand.

        Args:
            coords (tuple): (dims_x, dims_y), 
                          top left coordinates of image region at selected 
                          `read_mag` or `read_mpp` from `prepare_reading` 
            size (tuple): (dims_x, dims_y)
                          width and height of image region at selected 
                          `read_mag` or `read_mpp` from `prepare_reading`       

        """
        if self.image_ptr is None:
            # convert coord from read lv to lv zero
            lv_0_shape = np.array(self.file_ptr.level_dimensions[0])
            lv_r_shape = np.array(self.file_ptr.level_dimensions[self.read_lv])
            up_sample = (lv_0_shape / lv_r_shape)[0]
            new_coord = [0, 0]
            new_coord[0] = int(coords[0] * up_sample)
            new_coord[1] = int(coords[1] * up_sample)
            region = self.file_ptr.read_region(new_coord, self.read_lv, size)
        else:
            region = self.image_ptr[
                coords[1] : coords[1] + size[1], coords[0] : coords[0] + size[0]
            ]
        return np.array(region)[..., :3]

    def get_full_img(self, read_mag=None, read_mpp=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.

        `read_mpp` is in X, Y format.
        """

        read_lv, scale_factor = self._get_read_info(
            read_mag=read_mag, read_mpp=read_mpp
        )

        read_size = self.file_ptr.level_dimensions[read_lv]

        wsi_img = self.file_ptr.read_region((0, 0), read_lv, read_size)
        wsi_img = np.array(wsi_img)[..., :3]  # remove alpha channel
        if scale_factor is not None:
            # now rescale then return
            if scale_factor > 1.0:
                interp = cv2.INTER_CUBIC
            else:
                interp = cv2.INTER_LINEAR
            wsi_img = cv2.resize(
                wsi_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interp
            )
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

