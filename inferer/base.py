import argparse
import glob
import math
import multiprocessing
import os
import re
import sys
import warnings
warnings.filterwarnings('ignore') 
from importlib import import_module
from multiprocessing import Lock, Pool

import numpy as np
import torch
import torch.utils.data as data
import tqdm

####
class Inferer(object):
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
        model_desc = import_module('method_desc.%s.net_desc' % self.method['name'])
        model_creator = getattr(model_desc, 'create_model')

        # TODO: deal with parsing multi level model desc
        net = model_creator(**self.method['model_args'])
        net = torch.nn.DataParallel(net)
        saved_state_dict = torch.load(self.method['model_path'])
        # TODO: differentitate between dataparallel enclosed or not
        net.load_state_dict(saved_state_dict['desc'], strict=True)
        net = net.to('cuda')

        module_lib = import_module('method_desc.%s.run_desc' % self.method['name'])
        run_step = getattr(module_lib, 'infer_step')
        self.run_step = lambda input_batch : run_step(input_batch, net)

        module_lib = import_module('method_desc.%s.post_proc' % self.method['name'])
        self.post_proc_func = getattr(module_lib, 'process')
        return
