import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "hovernet"

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. If fast mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [164, 164] # patch shape at output of network

        self.dataset_name = "consep" # extracts dataset info from dataset.py
        self.log_dir = "logs/" # where checkpoints will be saved

        # paths to training and validation patches
        self.train_dir_list = [
            "/home/simon/Desktop/hover_net_pytorch/simon_patches"
        ]
        self.valid_dir_list = [
            "/home/simon/Desktop/hover_net_pytorch/simon_patches"
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)
        self.model_config_file = importlib.import_module(
            "models.%s.opt" % model_name
        )
