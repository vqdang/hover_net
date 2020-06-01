import importlib
import random

import cv2
import numpy as np

import dataset


class Config(object):
    """
    Configuration file.
    """

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False 

        self.type_classification = False  # whether to predict the nuclear type- dependent on dataset!

        # determines which dataset to be used during training / inference. The appropriate class
        # is initialised in dataset.py. Refer to dataset.py for info regarding data paths. Currently 
        # implemented: 'Kumar', 'CPM17', 'CoNSeP'. If using additional datasets, an appropriate 
        # class must be added in dataset.py
        self.dataset_name = 'Kumar' 

        self.log_dir = 'logs/tmp' # log directory where checkpoints are saved

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

        # dynamically set the config file into variable
        self.model_config_file = importlib.import_module('model.opt')
