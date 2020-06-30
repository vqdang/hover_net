
"""run_infer.py

Usage:
  run_infer.py [--gpu=<id>] [--name=<name>] [--mode=<mode>] [--nr_types=<n>] [--model_path=<path>] \
               [--nr_inference_workers=<n>] [--nr_post_proc_workers=<n>] [--batch_size=<n>] \
               [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] [--wsi_proc_mag=<n>] \
               [--cache_path=<path>] [--input_wsi_dir=<path>] [--input_msk_dir=<path>] \
               [--output_dir=<path>] [--patch_input_shape=<n>] [--patch_output_shape=<n>]
  run_infer.py (-h | --help)
  run_infer.py --version

Options:
  -h --help                   Show this string.
  --version                   Show version.
  --gpu=<id>                  GPU list. [default: 0]
  --name=<name>               Model name. [default: hovernet]
  --mode=<mode>               Inference mode. 'tile' or 'wsi'. [default: tile]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --model_path=<path>         Path to saved checkpoint.
  --nr_inference_workers=<n>  Number of workers during inference. [default: 4]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 4]
  --batch_size=<n>            Batch size. [default: 32]
  --ambiguous_size=<n>        Ambiguous size. [default: 128]
  --chunk_shape=<n>           Shape of chunk for processing. [default: 10000]
  --tile_shape=<n>            Shape of tiles for processing. [default: 4096]
  --wsi_proc_mag=<n>          Magnification level used for WSI processing. [default: -1]
  --cache_path=<path>         Path for cache. Should be placed on SSD with at least 100GB. [default: cache/]
  --input_wsi_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
  --input_msk_dir=<path>      Path to directory containing tissue masks. Should have the same name as corresponding WSIs.
  --output_dir=<path>         Path to output data directory. Will create automtically if doesn't exist. [default: output/]
  --patch_input_shape=<n>     Shape of input patch to the network- Assume square shape. [default: 270]
  --patch_output_shape=<n>    Shape of network output- Assume square shape. [default: 80]
"""

import os
from docopt import docopt

#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = docopt(__doc__, version='HoVer-Net Pytorch Inference v1.0')

    os.environ['CUDA_VISIBLE_DEVICES'] = args['--gpu']

    # raise exceptions for invalid / missing arguments
    if args['--model_path'] == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')
    if args['--mode'] != 'tile' and args['--mode'] != 'wsi':
        raise Exception('Mode not recognised. Use either "tile" or "wsi"')
    if args['--input_wsi_dir'] == None:
        raise Exception('An input directory must be supplied as an argument with --input_wsi_dir.')
    if args['--input_wsi_dir'] == args['--output_dir']:
        raise Exception('Input and output directories should not be the same- otherwise input directory will be overwritten.')
    nr_types = int(args['--nr_types'])
    if nr_types == 0:
        nr_types = None 

    ################### * SAMPLE CALL FOR TILE INFER
    # method_args = {
    #     'method' : {
    #         'name'       : 'micronet',
    #         'model_args' : {
    #             'nr_types'   : None,
    #         },
    #         'model_path' : 'exp_output/micronet/kumar/00/net_epoch=100.tar',
    #     },
    # }
    # # run_args = {
    # #     'nr_inference_workers' : 2,
    # #     'nr_post_proc_workers' : 2,
    # #     'batch_size' : 4,
    # #     'input_dir'  : '/home/tialab-dang/workspace/dataset/NUC_HE_Kumar/train-set/orig_split/valid_same/',
    # #     'output_dir' : 'exp_output/dump/',
    # #     'patch_input_shape'  : 252, # always be square RoI
    # #     'patch_output_shape' : 252, # always be square RoI
    # # }
    # from inferer.tile import Inferer
    # inferer = Inferer(**method_args)

    # run_args = {
    #     'nr_inference_workers' : 2,
    #     'nr_post_proc_workers' : 2,
    #     'batch_size' : 4,
    #     'input_dir'  : '/home/tialab-dang/workspace/dataset/NUC_HE_Kumar/train-set/orig_split/valid_diff/',
    #     'output_dir' : 'exp_output/dump/',
    #     'patch_input_shape'  : 252, # always be square RoI
    #     'patch_output_shape' : 252, # always be square RoI
    # }
    # from inferer.tile import Inferer

    # inferer.process_file_list(**run_args)
    ###################

    method_args = {
        'method' : {
            'name'       : args['--name'],
            'model_args' : {
                'nr_types'   : nr_types
            },
            'model_path' : args['--model_path'],
        },
    }
    run_args = {
        'nr_inference_workers' : int(args['--nr_inference_workers']),
        'nr_post_proc_workers' : int(args['--nr_inference_workers']),
        'batch_size'  : int(args['--batch_size']),

        'ambiguous_size' : int(args['--ambiguous_size']),
        'chunk_shape' : [int(args['--chunk_shape']),int(args['--chunk_shape'])],
        'tile_shape'  : [int(args['--tile_shape']),int(args['--tile_shape'])],
        'cache_path'  : args['--cache_path'], 

        'wsi_proc_mag'  : int(args['--wsi_proc_mag']), 
        'input_wsi_dir' : args['--input_wsi_dir'], 
        'input_msk_dir' : args['--input_msk_dir'], 
        'output_dir' : args['--output_dir'],
        'patch_input_shape'  : [int(args['--patch_input_shape']),int(args['--patch_input_shape'])], 
        'patch_output_shape' : [int(args['--patch_output_shape']),int(args['--patch_output_shape'])], 
    }

    #! Need to implement also for tile mode
    if args['--mode'] == 'wsi':
        from infer.wsi import InferManager
    else:
        assert False, "Unknown mode `%s`" % args['--mode']
    infer = InferManager(**method_args)
    infer.process_wsi_list(run_args)