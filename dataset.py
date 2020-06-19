import glob
import cv2
import numpy as np
import scipy.io as sio


####
class Kumar(object):
    """
    Defines the Kumar dataset as originally introduced in:

    Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane, 
    and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for 
    computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560.
    """
    def __init__(self):
        self.data_root = 'dataset'
        self.desc = {
            'train':
                {
                    'img': ('.tif', self.data_root + '/Kumar/train/Images/'),
                    'ann': ('.mat', self.data_root + '/Kumar/train/Labels/')
                },
            'valid_same':
                {
                    'img': ('.tif', self.data_root + '/Kumar/test_same/Images/'),
                    'ann': ('.mat', self.data_root + '/Kumar/test_same/Labels/')
                },
            'valid_diff':
                {
                    'img': ('.tif', self.data_root + '/Kumar/test_diff/Images/'),
                    'ann': ('.mat', self.data_root + '/Kumar/test_diff/Labels/')
                },
        }

        self.nr_types = None # no classification labels

        # used for determining the colour of contours in overlay
        self.type_colour = {
            0: (0, 0, 0),
            1: (255, 255, 0),
        }

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)['inst_map']
        ann_inst = ann_inst.astype('int32')
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class CPM17(object):
    """
    Defines the CPM 2017 dataset as originally introduced in:

    Vu, Quoc Dang, Simon Graham, Tahsin Kurc, Minh Nguyen Nhat To, Muhammad Shaban, 
    Talha Qaiser, Navid Alemi Koohbanani et al. "Methods for segmentation and classification 
    of digital microscopy tissue images." Frontiers in bioengineering and biotechnology 7 (2019).
    """
    def __init__(self):
        self.data_root = 'dataset'
        self.desc = {
            'train':
                {
                    'img': ('.png', self.data_root + '/cpm17/train/Images/'),
                    'ann': ('.mat', self.data_root + '/cpm17/train/Labels/')
                },
            'valid':
                {
                    'img': ('.png', self.data_root + '/cpm17/test/Images/'),
                    'ann': ('.mat', self.data_root + '/cpm17/test/Labels/')
                },
        }

        self.train_dir_list = [
            self.data_root + '/cpm17/patches/train/']
        self.valid_dir_list = [
            self.data_root + '/cpm17/patches/valid/']

        self.nr_types = None  # no classification labels

        # used for determining the colour of contours in overlay
        self.type_colour = {
            0: (0, 0, 0),
            1: (255, 255, 0),
        }


    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)['inst_map']
        ann_inst = ann_inst.astype('int32')
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class CoNSeP(object):
    """
    Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei 
    in multi-tissue histology images." Medical Image Analysis 58 (2019): 101563.
    """
    def __init__(self):
        self.data_root = 'dataset'
        self.desc = {
            'train':
                {
                    'img': ('.png', self.data_root + '/consep/Train/Images/'),
                    'ann': ('.mat', self.data_root + '/consep/Train/Labels/')
                },
            'valid':
                {
                    'img': ('.png', self.data_root + '/consep/Test/Images/'),
                    'ann': ('.mat', self.data_root + '/consep/Test/Labels/')
                },
        }

        self.nr_types = 5

        self.type_colour = {
            0: (  0,   0,   0),
            1: (255,   0,   0),
            2: (  0, 255,   0),
            3: (  0,   0, 255),
            4: (255, 255,   0),
            5: (255, 165,   0)
        }

        self.nuclei_type_dict = {
            'Miscellaneous': 1,
            'Inflammatory': 2,
            'Epithelial': 3,
            'Spindle': 4,
        }
        assert len(self.nuclei_type_dict.values()) == self.nr_types - 1

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)['inst_map']
        if with_type:
            ann_type = sio.loadmat(path)['type_map']

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype('int32')
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype('int32')

        return ann
