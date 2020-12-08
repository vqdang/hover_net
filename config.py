import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """
    Configuration file.
    """

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False 

        model_name = 'hovernet'

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        aug_shape = [540, 540]
        act_shape = [256, 256]
        out_shape = [164, 164]

        self.dataset_name = 'pannuke_full_v0' 
        self.log_dir = 'exp_output/models/hovernet/continual_data=[v0.0]_run=[v0.1]/'
        # self.log_dir = 'exp_output/models/dump/'

        self.train_dir_list = [
            # 'dataset/training_data/pannuke_full_v0/fold_1/540x540_164x164/',
            # 'dataset/training_data/pannuke_full_v0/fold_2/540x540_164x164/'

            'dataset/training_data/continual_v0.0/rmt/train/540x540_164x164'
        ]
        self.valid_dir_list = [
            # 'dataset/training_data/pannuke_full_v0/fold_3/540x540_164x164/'

            'dataset/training_data/continual_v0.0/rmt/valid/540x540_164x164'
        ]

        self.shape_info = {
            'train': {
                'input_shape': act_shape,
                'mask_shape' : out_shape,
            },
            'valid': {
                'input_shape': act_shape,
                'mask_shape' : out_shape,
            },
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)
        self.model_config_file = importlib.import_module('models.%s.opt_continual_v00' % model_name)
