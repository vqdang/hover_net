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
        self.type_classification = False

        # determines which dataset to be used during training / inference. The appropriate class
        # is initialised in dataset.py. Refer to dataset.py for info regarding data paths. Currently 
        # implemented: 'Kumar', 'CPM17', 'CoNSeP'. If using additional datasets, an appropriate 
        # class must be added in dataset.py
        self.dataset_name = 'CoNSeP' 

        # TODO:  self meta tagging and hash code for exp run
        # log directory where checkpoints are saved
        self.log_dir = 'exp_output/dump/'

        self.train_dir_list = [
            'dataset/train/consep/train/540x540_80x80/'
        ]
        self.valid_dir_list = [
            'dataset/train/consep/valid/540x540_80x80/'
        ]

        self.shape_info = {
            'train': {
                'input_shape': [270, 270],
                'mask_shape': [80, 80],
            },
            'valid': {
                'input_shape': [270, 270],
                'mask_shape': [80, 80],
            },
            'test': {
                'input_shape': [270, 270],
                'mask_shape': [80, 80],
            }
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)
        self.model_config_file = importlib.import_module('models.%s.opt' % model_name)
