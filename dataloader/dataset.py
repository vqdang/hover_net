
import glob
import cv2
import numpy as np

class Kumar(object):
    def __init__(self):
        self.img_ext = '.png'

        # TODO: instrinsic definition so hand-picked or ?
        self.desc = {
            'train'      : 
                {
                    'img' : ('.tif', '../../../dataset/NUC_HE_Kumar/train-set/orig_split/train/'),
                    'ann' : ('.npy', '../../../dataset/NUC_HE_Kumar/train-set/anns_proc/imgs_all/')
                },
            'valid_same' : 
                {
                    'img' : ('.tif', '../../../dataset/NUC_HE_Kumar/train-set/orig_split/valid_same/'),
                    'ann' : ('.npy', '../../../dataset/NUC_HE_Kumar/train-set/anns_proc/imgs_all/')
                },
            'valid_diff' :
                {
                    'img' : ('.tif', '../../../dataset/NUC_HE_Kumar/train-set/orig_split/valid_diff/'),
                    'ann' : ('.npy', '../../../dataset/NUC_HE_Kumar/train-set/anns_proc/imgs_all/')
                },
        }

        # TODO: unifiy at one place and dynamically populate here or ?
        self.train_data_dir = None
        self.valid_data_dir = None

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) 

    def load_ann(self, path):
        # assumes that ann is HxW
        # TODO: align with old github protocol !
        ann_inst = np.load(path)[...,1]
        ann_inst = ann_inst.astype('int32')
        ann = np.expand_dims(ann_inst, -1)
        return ann
