import warnings
warnings.filterwarnings('ignore') 

from multiprocessing import Pool, Lock
import multiprocessing
multiprocessing.set_start_method('spawn', True) # ! must be at top for VScode debugging
import argparse
import glob
from importlib import import_module
import math
import os
import sys
import re

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from docopt import docopt
import tqdm
import psutil
from dataloader.infer_loader import SerializeFileList, SerializeArray
from functools import reduce

from misc.patch_extractor import prepare_patching
from misc.utils import rm_n_mkdir, cropping_center, get_bounding_box
from postproc import hover

import openslide

####
class InferBase(object):
    def __init__(self, **kwargs):
        self.run_step = None
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        self.__load_model()
        self.type_classification = self.method['model_args']['nr_types'] is not None
        return

    def __load_model(self):
        """
        Create the model, load the checkpoint and define
        associated run steps to process each data batch
        """
        model_desc = import_module('models.%s.net_desc' % self.method['name'])
        model_creator = getattr(model_desc, 'create_model')

        # TODO: deal with parsing multi level model desc
        net = model_creator(**self.method['model_args'])
        saved_state_dict = torch.load(self.method['model_path'])
        net.load_state_dict(saved_state_dict['desc'], strict=True)
        net = torch.nn.DataParallel(net).to('cuda')

        run_desc = import_module('models.%s.run_desc' % self.method['name'])
        self.run_step = lambda input_batch : getattr(run_desc, 'infer_step')(input_batch, net)
        return
    