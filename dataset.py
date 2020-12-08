import glob
import cv2
import numpy as np
import scipy.io as sio


####
class __Kumar(object):
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
class __RMT(object):
    """
    Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei 
    in multi-tissue histology images." Medical Image Analysis 58 (2019): 101563.
    """
    def __init__(self, version_code=''):

        self.data_root = 'dataset/rmt/'
        self.desc = {
            'train':
                {
                    'img': ('.png', self.data_root + 'sample/imgs/'),
                    'ann': ('.npy', self.data_root + '%s/split/train/' % version_code)
                },
            'valid':
                {
                    'img': ('.png', self.data_root + 'sample/imgs/'),
                    'ann': ('.npy', self.data_root + '%s/split/valid/' % version_code)
                },
        }

        self.nr_types = 4

        self.type_colour = {
            0 : (0  ,   0,   0), 
            1 : (255,   0,   0), # neoplastic
            2 : (0  , 255,   0), # inflamm
            3 : (0  ,   0, 255), # connective
            4 : (255, 255,   0), # dead
            5 : (255, 165,   0), # non-neoplastic epithelial
        }

        self.nuclei_type_dict = {
            'Epithelium'     : 1,
            'Inflammatory'   : 2,
            'Connective'     : 3,
            'Dead'           : 4,
        }

    def load_img(self, path): # to ensure x40
        try:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)
        except:
            print('Error:', path)
            assert False
        return img

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxWx2
        ann = np.load(path) 
        # print(np.unique(ann[...,1]))
        if not with_type:
            ann = ann[...,:0]
            ann = np.expand_dims(ann, -1)
            ann = ann.astype('int32')
        ann = cv2.resize(ann, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        return ann

####
class __Pannuke(object):
    """
    Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei 
    in multi-tissue histology images." Medical Image Analysis 58 (2019): 101563.
    """
    def __init__(self):
        self.data_root = '../../dataset/pannuke_full/'
        self.desc = {
            'fold_1':
                {
                    'img': ('.jpg', self.data_root + 'fold_1/imgs/'),
                    'ann': ('.npy', self.data_root + 'fold_1/anns/')
                },
            'fold_2':
                {
                    'img': ('.jpg', self.data_root + 'fold_2/imgs/'),
                    'ann': ('.npy', self.data_root + 'fold_2/anns/')
                },
            'fold_3':
                {
                    'img': ('.jpg', self.data_root + 'fold_3/imgs/'),
                    'ann': ('.npy', self.data_root + 'fold_3/anns/')
                },
        }

        self.nr_types = 6

        self.type_colour = {
            0 : (0  ,   0,   0), 
            1 : (255,   0,   0), # neoplastic
            2 : (0  , 255,   0), # inflamm
            3 : (0  ,   0, 255), # connective
            4 : (255, 255,   0), # dead
            5 : (255, 165,   0), # non-neoplastic epithelial
        }

        self.nuclei_type_dict = {
            'Neoplastic'     : 1,
            'Inflammatory'   : 2,
            'Connective'     : 3,
            'Dead'           : 4,
            'Non-Neoplastic' : 5,
        }
        assert len(self.nuclei_type_dict.values()) == self.nr_types - 1

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann = np.load(path) 
        if not with_type:
            ann = ann[...,:0]
            ann = np.expand_dims(ann, -1)
            ann = ann.astype('int32')
        return ann

####
def get_dataset(name, **kwargs):
    """
    Return a pre-defined dataset object associated with `name`
    """
    if name.lower() == 'kumar':
        return __Kumar()
    if name.lower() == 'rmt':
        return __RMT(**kwargs)
    elif 'pannuke' in name.lower():
        return __Pannuke()
    else:
        assert False, "Unknown dataset `%s`" % name