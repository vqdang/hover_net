import glob
import cv2
import numpy as np
import scipy.io as sio


####
class Kumar(object):
    def __init__(self, type_classification=False):
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

        self.train_dir_list = [
            self.data_root + '/Kumar/patches/train/']
        self.valid_dir_list = [
            self.data_root + '/Kumar/patches/valid_same/',
            self.data_root + '/Kumar/patches/valid_diff/'
            ]

        # used for determining the colour of contours in overlay
        self.class_color = {
            0: (0, 0, 0),
            1: (255, 255, 0),
        }

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)['inst_map']
        ann_inst = ann_inst.astype('int32')
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class CPM17(object):
    def __init__(self, type_classification=False):
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

        # used for determining the colour of contours in overlay
        self.class_color = {
            0: (0, 0, 0),
            1: (255, 255, 0),
        }


    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)['inst_map']
        ann_inst = ann_inst.astype('int32')
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class CoNSeP(object):
    def __init__(self, type_classification=False):
        self.data_root = 'dataset'
        self.desc = {
            'train':
                {
                    'img': ('.png', self.data_root + '/CoNSeP/train/Images/'),
                    'ann': ('.mat', self.data_root + '/CoNSeP/train/Labels/')
                },
            'valid':
                {
                    'img': ('.png', self.data_root + '/CoNSeP/test/Images/'),
                    'ann': ('.mat', self.data_root + '/CoNSeP/test/Labels/')
                },
        }

        self.train_dir_list = [
            self.data_root + 'CoNSeP/patches/train/']
        self.valid_dir_list = [
            self.data_root + 'CoNSeP/patches/valid/']

        self.nr_types = 5

        if type_classification:
            # used for determining the colour of contours in overlay
            self.class_color = {
                0: (0, 0, 0),
                1: (255, 0, 0),
                2: (0, 255, 0),
                3: (0, 0, 255),
                4: (255, 255, 0),
                5: (255, 165, 0)
            }
        else:
            # used for determining the colour of contours in overlay
            self.class_color = {
                0: (0, 0, 0),
                1: (255, 255, 0),
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

    def load_ann(self, path):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)['inst_map']
        if self.type_classification:
            ann_type = sio.loadmat(path)['type_map']

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            assert np.max(ann_type) <= self.nr_types-1, \
                "Only %d types of nuclei are defined for training"\
                "but there are %d types found in the input image." % (
                    self.type_classification.nr_types, np.max(ann_type))

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype('int32')
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype('int32')

            self.class_color = {
                0: (0, 0, 0),
                1: (255, 255, 0),
            }
        return ann
