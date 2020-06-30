from collections import OrderedDict
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage import color

import openslide
import glymur


class FileHandler(object):
    def __init__(self):
        """
        The handler is responsible for storing the processed data, parsing
        the metadata from original file, and reading it from storage. 
        """
        self.logger = None
        self.mid_output_path = None

        self.metadata = {
            ('magnification', None),
            ('base_mag'     , None),
            ('vendor'       , None),
            ('mpp-x'        , None), # scan resolution
            ('mpp-y'        , None), # scan resolution
            ('height'       , None),
            ('width'        , None),
            ('levels'       , None), 
            ('level_dims'   , None)
        }

        self.data = {
            'rgb' : {
                'img'  : None,
                'gray' : None,
                'mask' : None,
            },

            'stat' : OrderedDict()
        }
        self.intermediate_output = []
        pass

    def __load_metadata(self):
        pass

    def read_region(self):
        pass

    def get_dimensions(self, level):
        pass

    def add_data_holder(self, name, data_dict=None):
        if data_dict is not None:
            self.data[name] = data_dict
        else:
            self.data[name] = {
                'img' : None,
                'gray': None,
                'mask': None,
            }
        return


class JP2Handler(FileHandler):
    """
    Class for handling JP2 whole-slide images.
    
    Note, JP2 WSIs use JPEG2000 compression and therefore are not stored in 
    an image pyramid. Nonetheless, we handle the image in such a way that 
    it can be accessed with similar commands as openslide supported WSIs.

    We will fix the number or psudo pyramid levels to be 5
    """
    def __init__(self, file_path):
        """
        file_path (string): path to single whole-slide image
        """
        super().__init__()
        self.file_ptr = glymur.Jp2k(file_path)
        self.metadata = self.__load_metadata()
        thumbnail = self.__load_thumbnail(1.25)

        data_dict = {
            'img'  : thumbnail,
            'gray' : color.rgb2gray(thumbnail), # thumbnail_gray
            'mask' : np.ones(thumbnail.shape[:2], dtype=np.bool)
        }
        self.add_data_holder('src', data_dict)


    def __load_metadata(self):
        metadata = {}
        
        level_0_magnification = 40.0
        box = self.file_ptr.box[2].box[0]
        dimensions = (box.width, box.height)

        downsample_level = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        magnification_level = [level_0_magnification / lv for lv in downsample_level]
        self.level_dimensions = [(dimensions[0], dimensions[1]),
                                (int(dimensions[0]/2), int(dimensions[1]/2)),
                                (int(dimensions[0]/4), int(dimensions[1]/4)),
                                (int(dimensions[0]/8), int(dimensions[1]/8)),
                                (int(dimensions[0]/16), int(dimensions[1]/16)),
                                (int(dimensions[0]/32), int(dimensions[1]/32))
                                ]

        metadata = [
            ('magnification', magnification_level),
            ('base_mag'     , magnification_level[0]),
            ('vendor'       , None),
            ('mpp-x'        , 0.275), # scan resolution of an Omnyx scanner at UHCW
            ('mpp-y'        , 0.275), # scan resolution of an Omnyx scanner at UHCW
            ('height'       , self.level_dimensions[0][0]),
            ('width'        , self.level_dimensions[0][1]),
            ('levels'       , 6), # fix 6 levels,
            ('level_dims'   , self.level_dimensions)
        ]
        return OrderedDict(metadata)
    
    def read_region(self, coords, read_level, read_level_size):
        """
        read a region from a JP2 WSI

        Args:
            coords (tuple): top left coordinates of image region at level 0 (x,y)
            read_level (int): level of image pyramid to read from
            read_level_size (tuple): dimensions of image region at selected level (dims_x, dims_y)
        """
        factor = 2**read_level # indexing is at 40x
        return self.file_ptr[coords[1]:coords[1]+read_level_size[1]*factor:factor,
                        coords[0]:coords[0]+read_level_size[0]*factor:factor,:]
        
    def __load_thumbnail(self, magnification, read_level=2):
        # width-height, not height-width
        read_level_size = self.metadata['level_dims'][read_level]
        read_level_magnification = self.metadata['magnification'][read_level]
        img = self.read_region((0, 0), read_level, read_level_size)
        
        img_low_res = cv2.resize(img, (0, 0), 
                fx=magnification/read_level_magnification,
                fy=magnification/read_level_magnification,
                interpolation=cv2.INTER_AREA)

        return img_low_res
    
    def get_dimensions(self, level):
        return self.level_dimensions[level]


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
        thumbnail = self.__load_thumbnail(1.25)

        data_dict = {
            'img'  : thumbnail,
            'gray' : color.rgb2gray(thumbnail), # thumbnail_gray
            'mask' : np.ones(thumbnail.shape[:2], dtype=np.bool)
        }
        self.add_data_holder('src', data_dict)

    def __load_metadata(self):
        metadata = {}
        
        wsi_properties = self.file_ptr.properties
        level_0_magnification = wsi_properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        level_0_magnification = float(level_0_magnification)

        downsample_level = self.file_ptr.level_downsamples
        magnification_level = [level_0_magnification / lv for lv in downsample_level]

        metadata = [
            ('magnification', magnification_level),
            ('base_mag'     , magnification_level[0]),
            ('vendor'       , wsi_properties[openslide.PROPERTY_NAME_VENDOR]),
            ('mpp-x'        , wsi_properties[openslide.PROPERTY_NAME_MPP_X]),
            ('mpp-y'        , wsi_properties[openslide.PROPERTY_NAME_MPP_Y]),
            ('height'       , self.file_ptr.dimensions[1]),
            ('width'        , self.file_ptr.dimensions[0]),
            ('levels'       , self.file_ptr.level_count),
            ('level_dims'  , self.file_ptr.level_dimensions)
        ]
        return OrderedDict(metadata)
    
    def read_region(self, coords, read_level, read_level_size):
        """
        read a region from openslide object

        Args:
            coords (tuple): top left coordinates of image region at level 0 (x,y)
            read_level (int): level of image pyramid to read from
            read_level_size (tuple): dimensions of image region at selected level (dims_x, dims_y)
        """
        region = self.file_ptr.read_region(coords, read_level, read_level_size)
        return np.array(region)[...,:3]
        
    def __load_thumbnail(self, magnification, read_level=2):
        """
        load a thumbnail from openslide object

        Args:
            magnification (float): objective magnification
            read_level (int): level of pyramid to read from
        """
        # width-height, not height-width
        read_level_size = self.metadata['level_dims'][read_level]
        read_level_magnification = self.metadata['magnification'][read_level]
        img = self.read_region((0, 0), read_level, read_level_size)
        
        img_low_res = cv2.resize(img, (0, 0), 
                fx=magnification/read_level_magnification,
                fy=magnification/read_level_magnification,
                interpolation=cv2.INTER_AREA)

        return img_low_res
    
    def get_dimensions(self, level):
        return self.file_ptr.level_dimensions[level]


def get_file_handler(path, backend):
    if backend in ['svs','ndpi']:
        return OpenSlideHandler(path)
    elif backend == 'jp2':
        return JP2Handler(path)
    else:
        assert False, "Unknown WSI format `%s`" % backend