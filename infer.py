
"""infer.py

Usage:
  infer.py [--gpu=<id>] [--mode=<mode>] [--model=<path>] [--batch_size=<n>] [--num_workers=<n>] [--input_dir=<path>] [--output_dir=<path>] [--tile_size=<n>] [--return_masks]
  infer.py (-h | --help)
  infer.py --version

Options:
  -h --help            Show this string.
  --version            Show version.
  --gpu=<id>           GPU list. [default: 0]
  --mode=<mode>        Inference mode. 'tile' or 'wsi'. [default: tile]
  --model=<path>       Path to saved checkpoint.
  --input_dir=<path>   Directory containing input images/WSIs.
  --output_dir=<path>  Directory where the output will be saved. [default: output/]
  --batch_size=<n>     Batch size. [default: 25]
  --num_workers=<n>    Number of workers. [default: 12]
  --tile_size=<n>      Size of tiles (assumes square shape). [default: 20000]
  --return_masks       Whether to return cropped nuclei masks
"""
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

from docopt import docopt

#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = docopt(__doc__, version='HoVer-Net Pytorch Inference v1.0')

    # os.environ['CUDA_VISIBLE_DEVICES'] = args['--gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # raise exceptions for invalid / missing arguments
    # if args['--model'] == None:
    #     raise Exception('A model path must be supplied as an argument with --model.')
    # if args['--mode'] != 'tile' and args['--mode'] != 'wsi':
    #     raise Exception('Mode not recognised. Use either "tile" or "wsi"')
    # if args['--input_dir'] == None:
    #     raise Exception('An input directory must be supplied as an argument with --input_dir.')
    # if args['--input_dir'] == args['--output_dir']:
    #     raise Exception('Input and output directories should not be the same- otherwise input directory will be overwritten.')

    # import libraries for WSI processing
    # if args['--mode'] == 'wsi':
    #     import openslide as ops
    #     import glymur
    
    # method_args = {
    #     'method' : {
    #         'name'       : 'hovernet',
    #         'model_args' : {
    #             'nr_types'   : None,
    #         },
    #         'model_path' : 'exp_output/consep/bce+dice+mse+msge_v1/01/net_epoch=36.tar',
    #     },
    # }
    # run_args = {
    #     'nr_worker'  : 4,
    #     'batch_size' : 16,
    #     'input_dir'  : 'dataset/consep/Test/Images/',
    #     'output_dir' : 'exp_output/dump/',
    #     'patch_input_shape'  : 270, # always be square RoI
    #     'patch_output_shape' :  80, # always be square RoI
    #     'save_intermediate_output' : True,        
    # }
    # # if args['--mode'] == 'tile':
    # inferer = InfererTile(**method_args)
    # inferer.process_file_list(**run_args)

    method_args = {
        'method' : {
            'name'       : 'hovernet',
            'model_args' : {
                'nr_types'   : None,
            },
            'model_path' : 'dataset/home/net_epoch=50.tar',
        },
    }
    run_args = {
        'nr_worker'  : 4,
        'batch_size' : 32,
        'patch_input_shape'  : [270, 270], # always be square RoI
        'patch_output_shape' : [80, 80], # always be square RoI
        'save_intermediate_output' : True,        
    }
    from inferer.wsi import Inferer
    inferer = Inferer(**method_args)
    inferer.process_single_file()


# * OVERLAPPING TEST CAN REMOVE THE BOUNDARY PROBLEM ! HURRAH