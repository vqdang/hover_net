
import json
import glob

import operator
import os
import shutil

import numpy as np


####
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)

####
def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [rmin, rmax, cmin, cmax]

####
def cropping_center(x, crop_shape, batch=False):   
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:,h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]        
    return x

####
def rm_n_mkdir(dir_path):
    if (os.path.isdir(dir_path)):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

####
def get_files(data_dir_list, data_ext):
    """
    Given a list of directories containing data with extention 'date_ext',
    generate a list of paths for all files within these directories
    """

    data_files = []
    for sub_dir in data_dir_list:
        files = glob.glob(sub_dir + '/*'+ data_ext)
        data_files.extend(files)

    return data_files

####
def get_best_chkpts(path, metric_name, comparator='>'):
    """
    Return the path to checkpoint having largest/smallest measurement 
    values

    Args:
        path       :  chkpts directory, must contain "stats.json" file
        metric_name:  name of the metric within "stats.json" such as
                      'valid_acc' or 'valid_dice' etc,
        comparator :  '>' or '<'
    """
    stat_file_path = path + '/stats.json'
    ops = {
            '>': operator.gt,
            '<': operator.lt,
          }

    op_func = ops[comparator]
    with open(stat_file_path) as stat_file:
        info = json.load(stat_file)
    
    if comparator == '>':
        best_value  = -float("inf")
    else:
        best_value  = +float("inf")

    best_chkpts = 0
    for epoch_stat in info:
        epoch_value = epoch_stat[metric_name]
        if op_func(epoch_value, best_value):
            best_value  = epoch_value
            best_chkpts = epoch_stat['global_step']
    best_chkpts = "%smodel-%d.index" % (path, best_chkpts)
    return best_chkpts
