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

import json
import numpy as np
import torch
import torch.utils.data as data
import tqdm

####
class InferManager(object):
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
        model_desc = import_module('models.%s.net_desc' % self.method['model_name'])
        model_creator = getattr(model_desc, 'create_model')

        # TODO: deal with parsing multi level model desc
        net = model_creator(**self.method['model_args'])
        saved_state_dict = torch.load(self.method['model_path'])
        if list(saved_state_dict['desc'].keys())[0].split('.')[0] == 'module':
            net = torch.nn.DataParallel(net)
            net.load_state_dict(saved_state_dict['desc'], strict=True)
        else:
            net.load_state_dict(saved_state_dict['desc'], strict=True)
            net = torch.nn.DataParallel(net)
        net = net.to('cuda')
    
        module_lib = import_module('models.%s.run_desc' % self.method['model_name'])
        run_step = getattr(module_lib, 'infer_step')
        self.run_step = lambda input_batch : run_step(input_batch, net)

        module_lib = import_module('models.%s.post_proc' % self.method['model_name'])
        self.post_proc_func = getattr(module_lib, 'process')
        return

    def __jsonify_dict(self, path, old_dict, mag=None):
        new_dict = {}
        for inst_id, inst_info in old_dict.items():
            new_inst_info = {}
            for info_name, info_value in inst_info.items():
                # convert to jsonable
                if isinstance(info_value, np.ndarray):
                    info_value = info_value.tolist()
                new_inst_info[info_name] = info_value
            new_dict[int(inst_id)] = new_inst_info

        json_dict = { # to sync the format protocol
            'mag' : mag,
            'nuc' : new_dict
        }
        with open(path, 'w') as handle:
            json.dump(json_dict, handle)
        return new_dict