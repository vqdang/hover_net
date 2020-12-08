from collections import OrderedDict
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage import color
import re
import subprocess

import tifffile
import openslide
import glymur

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
            ('mpp  '        , None),
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

    def prepare_reading(self, read_mag=None, read_mpp=None, cache_path=None):
        """
        Only use either of these parameter, prioritize `read_mpp`

        `read_mpp` is in X, Y format
        """
        read_lv, scale_factor = self._get_read_info(
                                        read_mag=read_mag, 
                                        read_mpp=read_mpp)  

        if scale_factor is None:
            self.image_ptr = None
            self.read_lv = read_lv
        else:
            np.save(cache_path, self.get_full_img(read_mag=read_mag))
            self.image_ptr = np.load(cache_path, mmap_mode='r')      
        return

    def _get_read_info(self, read_mag=None, read_mpp=None):
        if read_mpp is not None:
            assert read_mpp[0] == read_mpp[1], 'Not supported uneven `read_mpp`'
            read_scale = (self.metadata['base_mpp'] / read_mpp)[0]
            read_mag = read_scale * self.metadata['base_mag']

        hires_mag = read_mag
        scale_factor = None
        if read_mag not in self.metadata['available_mag']:
            if read_mag > self.metadata['base_mag']:
                scale_factor = read_mag / self.metadata['base_mag']
                hires_mag = self.metadata['base_mag']
            else:          
                mag_list = np.array(self.metadata['available_mag'])
                mag_list = np.sort(mag_list)[::-1]
                hires_mag = mag_list - read_mag
                # only use higher mag as base for loading
                hires_mag = hires_mag[hires_mag > 0]
                # use the immediate higher to save compuration
                hires_mag = mag_list[np.argmin(hires_mag)]
                scale_factor = read_mag / hires_mag

        hires_lv = self.metadata['available_mag'].index(hires_mag) 
        return hires_lv, scale_factor

class JP2Handler(FileHandler):
    """
    Class for handling JP2 whole-slide images.
    
    Note, JP2 WSIs use JPEG2000 compression and therefore are not stored in 
    an image pyramid. Nonetheless, we handle the image in such a way that 
    it can be accessed with similar commands as openslide supported WSIs.

    We will fix the number or pseudo pyramid levels to be 6
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
            'img'  : thumbnail
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

        # only used for cases where the read magnification is different from
        self.image_ptr = None # the existing modes of the read file
        self.read_level = None

    def __load_metadata(self):
        metadata = {}
        
        wsi_properties = self.file_ptr.properties
        level_0_magnification = wsi_properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        level_0_magnification = float(level_0_magnification)

        downsample_level = self.file_ptr.level_downsamples
        magnification_level = [level_0_magnification / lv for lv in downsample_level]

        mpp = [wsi_properties[openslide.PROPERTY_NAME_MPP_X],
               wsi_properties[openslide.PROPERTY_NAME_MPP_Y]]
        mpp = np.array(mpp)

        metadata = [
            ('available_mag', magnification_level), # highest to lowest mag
            ('base_mag'     , magnification_level[0]),
            ('vendor'       , wsi_properties[openslide.PROPERTY_NAME_VENDOR]),
            ('mpp  '        , mpp),
            ('base_shape'   , np.array(self.file_ptr.dimensions)),
        ]
        return OrderedDict(metadata)
    
    def read_region(self, coords, size):
        """
        Must call `prepare_image` before hand

        Args:
            coords (tuple): top left coordinates of image region at level 0 (x,y)
            read_level (int): level of image pyramid to read from
            read_level_size (tuple): dimensions of image region at selected level (dims_x, dims_y)
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
            region = self.image_ptr[coords[1]:coords[1]+size[1],
                                    coords[0]:coords[0]+size[0]]
        return np.array(region)[...,:3]

    def get_full_img(self, read_mag=None, read_mpp=None):

        read_lv, scale_factor = self._get_read_info(
                                        read_mag=read_mag, 
                                        read_mpp=read_mpp)        

        read_size = self.file_ptr.level_dimensions[read_lv]

        wsi_img = self.file_ptr.read_region((0, 0), read_lv, read_size)
        wsi_img = np.array(wsi_img)[...,:3] # remove alpha channel
        if scale_factor is not None:
            # now rescale then return
            if scale_factor > 1.0:
                interp = cv2.INTER_CUBIC
            else:
                interp = cv2.INTER_LINEAR
            wsi_img = cv2.resize(wsi_img, (0, 0), 
                            fx=scale_factor, fy=scale_factor,
                            interpolation=interp)
        return wsi_img         

class QptiffHandler(FileHandler):
    """
    Because the arrangement of Tiff metada within qptiff look so random compared
    to .svs and .ndpi . Current attempt spit out dimension etc.

    Such as, the tiff page doesn't seem to be saved according to highest-lowest 
    resolution, so access to .qptiff should be through the page index, not 
    level index like .svs
    """
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.metadata = self.__load_metadata(file_path)

        # only used for cases where the read magnification is different from
        self.image_ptr = None # the existing modes of the read file
        self.read_level = None

    def __load_metadata(self, file_path):
        metadata = []

        float_regex = r'[-+]?\d*\.\d+|\d+'
        parse_width  = lambda x : re.findall('Image Width: (%s)'  % float_regex, x)
        parse_height = lambda x : re.findall('Image Length: (%s)' % float_regex, x)
        parse_resolution = lambda x : re.findall('Resolution: (%s), (%s) (.*?)\n' % \
                                                    (float_regex, float_regex), x)

        # * run external command to extract the header from binary
        raw_metadata = subprocess.run(['tiffinfo', '%s' % file_path], 
                    stdout=subprocess.PIPE, universal_newlines=True)
        raw_metadata = raw_metadata.stdout

        # * split head metadata according to Tiff directory header
        tiff_dir_idx = [m.start() for m in re.finditer('TIFF Directory', raw_metadata)] + [-1,]
        tiff_dir_raw_metdata_list = []
        for idx in range(len(tiff_dir_idx)-1):
            start_idx = tiff_dir_idx[idx]
            end_idx = tiff_dir_idx[idx+1]
            tiff_dir_raw_metdata = raw_metadata[start_idx:end_idx]
            tiff_dir_raw_metdata_list.append(tiff_dir_raw_metdata)

        acquisition = re.findall('<AcquisitionSoftware>(.*?)</AcquisitionSoftware>', raw_metadata)
        metadata.append(('vendor', list(set(acquisition))))

        magnification = re.findall('<Objective>(%s)x</Objective>' % float_regex, raw_metadata)
        magnification = [int(v) for v in magnification]
        base_mag = max(list(set(magnification)))

        image_type = re.findall('<Mode>(.*?)</Mode>', raw_metadata)
        image_type = list(set(image_type)) # should only be of 1 type
        assert len(image_type) == 1
        image_type = image_type[0]

        if image_type == 'im_Brightfield':
            metadata.append(('type', 'Bright Field'))
            # * assuming find all preserve search order
            # Brightfield usually is in RGB and each .tif dir corresponds to 1 resolution,
            # so just take the fisrt tiff dir (usually is the full resolution) to check for size etc.
            tiff_width_list  = parse_width(raw_metadata)
            tiff_height_list = parse_height(raw_metadata)
            tiff_width_list  = [int(v) for v in tiff_width_list]
            tiff_height_list = [int(v) for v in tiff_height_list]
            tiff_size_list = list(zip(tiff_height_list, tiff_width_list))

            resolution = parse_resolution(raw_metadata)

            # TODO: retrace mag and the like using same method as IF qptiff
            mppx = []
            mppy = []
            for i in range(len(resolution)): # TODO: width heigh or height width?
                if resolution[i][-1] == 'pixels/cm':
                    mppx.append(1.0e4 / float(resolution[i][0]))
                    mppy.append(1.0e4 / float(resolution[i][1]))
                else:
                    mppx.append('N/A')
                    mppy.append('N/A')

            # to get correct page by page magnification, get 
            # the resolution of first page then divide it per other level
            lv_mag = [base_mag * mppx[0] / mppx_lv for mppx_lv in mppx]

            lv_metadata_list = []
            for lv_idx in range(len(tiff_size_list)):
                lv_metadata = [
                    ('mag'         , lv_mag[lv_idx]),
                    ('shape'       , np.array(tiff_size_list[lv_idx])[::-1]),
                    ('mpp'         , np.array([mppx[lv_idx], mppy[lv_idx]])),
                    ('dir_idx_list', [lv_idx]),
                    ('stain_list'  , None)
                ]               
                lv_metadata_list.append(OrderedDict(lv_metadata))
            # cant be more specific becausse meaning of tag level is unclear
            lv_metadata_list = sorted(lv_metadata_list, key=lambda x: x['mag'])[::-1]
            metadata.append(('level_info', lv_metadata_list))
            metadata = OrderedDict(metadata)
        elif image_type == 'im_Fluorescence':
            metadata.append(('type', 'Fluorescence'))
            # TODO: later
            # * assuming find all preserve search order
            # have to group tiff dir according to their resolution to reliably 
            # extract number of channels or number of available levels

            # TODO: different name to differentiate raw metadata and parsed metadata
            # ! group according to resolution may not be enough
            tiff_width_list  = re.findall('Image Width: (%s)' % float_regex, raw_metadata)
            tiff_height_list = re.findall('Image Length: (%s)' % float_regex, raw_metadata)
            tiff_width_list  = [int(v) for v in tiff_width_list]
            tiff_height_list = [int(v) for v in tiff_height_list]
            tiff_size_list = list(zip(tiff_width_list, tiff_height_list))
            tiff_size_list = list(set(tiff_size_list))

            # extract resolution of each tiff dir to group it
            tiff_size_list = sorted(tiff_size_list, key=lambda x: x[0], reverse=True)
            lv_metadata_dict = []
            for tiff_size in tiff_size_list:
                lv_init_metadata = [
                    ('shape'        , np.array(tiff_size)), # W x H
                    ('mpp'          , None),
                    ('stain_list'   , []  ),
                    ('dir_idx_list' , []  )
                ]
                lv_init_metadata_dict = OrderedDict(lv_init_metadata)
                lv_metadata_dict.append((tiff_size, lv_init_metadata_dict))
            lv_metadata_dict = OrderedDict(lv_metadata_dict)

            base_size = np.array(tiff_size_list[0])
            for dir_idx, dir_raw_metadata in enumerate(tiff_dir_raw_metdata_list):
                dir_width  = parse_width(dir_raw_metadata)[0]
                dir_height = parse_height(dir_raw_metadata)[0]
                dir_size = np.array([int(dir_width), int(dir_height)])

                stain = re.findall('<Responsivity>(.*?)</Responsivity>', dir_raw_metadata, flags=re.DOTALL)
                if len(stain) == 1:
                    stain = re.findall('<Name>(.*?)</Name>', stain[0], flags=re.DOTALL)[0]
                else:
                    stain = "#CANT_PARSE_STAIN"
                
                lv_metadata = lv_metadata_dict[tuple(dir_size.tolist())]
                lv_metadata['stain_list'].append(stain)
                lv_metadata['dir_idx_list'].append(dir_idx)

                # * <Objective> tag is unreliable, likely is set basing on the highest
                # * hence, must use the resolution field to calculate back the magnification
                # do we need to cross verify across page?
                dir_res = parse_resolution(dir_raw_metadata)[0]
                if dir_res[-1] == 'pixels/cm':
                    dir_mpp = np.array([1.0e4 / float(dir_res[0]),
                                        1.0e4 / float(dir_res[1])]) # x, y
                else:
                    dir_mpp = np.array([None, None])
                    
                if lv_metadata['mpp'] is None:
                    lv_metadata['mpp'] = dir_mpp
                if np.any(lv_metadata['mpp'] != dir_mpp):
                    print('Fail Resolution')
            
            lv_metadata_list = list(lv_metadata_dict.values())
            lv_metadata_list = sorted(lv_metadata_list, key=lambda x: x['mpp'][0])
            base_mpp = lv_metadata_list[0]['mpp']
            for lv_metadata in lv_metadata_list:
                lv_mag = (base_mpp / lv_metadata['mpp']) * base_mag
                if np.unique(lv_mag).shape[0] > 1:
                    print('Uneven Mag Height Width')
                lv_metadata['mag'] = round(lv_mag[0], 2)
            base_stain_list = set(lv_metadata_list[0]['stain_list'])

            lv_metadata_list = [lv for lv in lv_metadata_list \
                                if set(lv['stain_list']) == base_stain_list]
            metadata.append(('level_info', lv_metadata_list))
            metadata = OrderedDict(metadata)

        self.raw_metadata = metadata
        simplified_metadata = [
            ('available_mag', [lv['mag'] for lv in lv_metadata_list]), # highest to lowest mag
            ('base_mag'     , lv_metadata_list[0]['mag']),
            ('base_mpp'     , lv_metadata_list[0]['mpp']),
            ('base_shape'   , lv_metadata_list[0]['shape']),
            ('vendor'       , metadata['vendor']),
        ]
        return OrderedDict(simplified_metadata)

    def read_region(self, coords, size):
        """
        Must call `prepare_image` before hand

        Args:
            coords (tuple): top left coordinates of image region at level 0 (x,y)
            read_level (int): level of image pyramid to read from
            read_level_size (tuple): dimensions of image region at selected level (dims_x, dims_y)
        """
        region = self.image_ptr[coords[1]:coords[1]+size[1],
                                coords[0]:coords[0]+size[0]]
        return np.array(region)

    def get_full_img(self, read_mag=None, read_mpp=None):

        read_lv, scale_factor = self._get_read_info(read_mag=read_mag, 
                                                            read_mpp=read_mpp)        
        hires_lv_info = self.raw_metadata["level_info"][read_lv]

        # list of separate IHC channels
        ch_list = [tifffile.imread(self.file_path, key=page_idx)
                        for page_idx in hires_lv_info['dir_idx_list']]

        if scale_factor is not None:            
            for ch_idx, ch in enumerate(ch_list):
                rz_ch = cv2.resize(ch, (0, 0), # * resize down
                    fx=scale_factor, fy=scale_factor,
                    interpolation=cv2.INTER_AREA)
                ch_list[ch_idx] = rz_ch
        ch_list = np.dstack(ch_list)

        return ch_list

def get_file_handler(path, backend):
    if backend in ['svs','ndpi']:
        return OpenSlideHandler(path)
    elif backend == 'jp2':
        return JP2Handler(path)
    elif backend == 'qptiff':
        return QptiffHandler(path)
    else:
        assert False, "Unknown WSI format `%s`" % backend

if __name__ == '__main__':
    path = '/mnt/user-temp/tialab-dang/dataset/RMT_mIHC_HE_Correlation/v0_scan/HE_Scans/NO3804012.qptiff'
    cache_path = 'dataset/cache/sample.npy'
    handler = get_file_handler(path, 'qptiff')
    handler.prepare_reading(read_mag=20.0, cache_path=cache_path)
    sample = handler.read_region([10000, 5000], [5000, 5000])
    import matplotlib.pyplot as plt
    plt.imshow(sample)
    plt.savefig('dump.png')
    print('here')
