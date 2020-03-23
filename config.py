import importlib
import random

import cv2
import numpy as np

#### 
class Config(object):
    def __init__(self, ):
        self.seed = 10 
        
        self.logging = True
        self.debug = False # turn on debug flag so that can trace some parallel processing problems easier
        self.type_classification = False # whether to predict the nuclear type
        # ! must use CoNSeP dataset, where nuclear type labels are available

        # TODO: the number of type and classes should be hardcoded into the dataset definition
        # denotes number of classes for nuclear type classification, 
        # plus the background class
        self.nr_types = 5
        # ! some semantic segmentation network like micronet,
        # ! nr_types will replace nr_classes if type_classification=True
        self.nr_classes = 2 # Nuclei Pixels vs Background

        # list of directories containing validation patches. 
        # For both train and valid directories, a comma separated list of directories can be used
        self.train_dir_list = ['dataset/train//Kumar/train/540x540_80x80/']
        self.valid_dir_list = ['dataset/train//Kumar/train/540x540_80x80/']

        self.shape_info = {
            'train' : {
                'input_shape' : [270, 270],
                'mask_shape'  : [80, 80],
                # 'mask_shape'  : [270, 270],
            },      
            'valid' : {
                'input_shape' : [270, 270],
                'mask_shape'  : [80, 80],
            }      
        }

        #### Dynamically setting the config file into variable
        self.model_config_file = importlib.import_module('model.opt')

        # for variable, value in config_dict.items():
        #     self.__setattr__(variable, value)
        ####

        # define your nuclei type name here, please ensure it contains
        # same the amount as defined in `self.nr_types` . ID 0 is preserved
        # for background so please don't use it as ID
        self.nuclei_type_dict = {
            'Miscellaneous': 1, # ! Please ensure the matching ID is unique
            'Inflammatory' : 2,
            'Epithelial'   : 3,
            'Spindle'      : 4,
        }
        assert len(self.nuclei_type_dict.values()) == self.nr_types - 1
