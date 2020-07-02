
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
import copy
from docopt import docopt

#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = docopt(__doc__, version='HoVer-Net Pytorch Inference v1.0')

    if args['--gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['--gpu']

    # raise exceptions for invalid / missing arguments
    if args['--run-mode'] != 'tile' and args['--mode'] != 'wsi':
        raise Exception('Mode not recognised. Use either "tile" or "wsi"')

    # TODO: exposed model kwargs ?
    if args['--method-model_path'] == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')
    if args['--method-name'] == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')
    nr_types = int(args['--nr_types']) if nr_types > 0 else None
    method_args = {
        'method' : {
            'name'       : args['--name'],
            'model_args' : {
                'nr_types'   : nr_types
            },
            'model_path' : args['--model_path'],
        },
    }

    if args['--run-mode'] == 'tile':
        default_run_args = {
            'nr_inference_workers' : 4,
            'nr_post_proc_workers' : 16,
            'batch_size' : 4,
            'tile_input_dir'  : None,
            'tile_output_dir' : None,
            'patch_input_shape'  : None, # always be square RoI
            'patch_output_shape' : None, # always be square RoI
        }
    else:
        default_run_args = {
            'nr_inference_workers' : 4,
            'nr_post_proc_workers' : 16,
            'batch_size'  : 4,

            'ambiguous_size' : 128,
            'chunk_shape' : 10000,
            'tile_shape'  : 2048,
            'wsi_cache_path' : '', 

            'wsi_proc_mag'  : -1, # -1 default to highest
            'input_wsi_dir' : None, # asumme no nested dir
            'input_msk_dir' : None, # should have the same name as one within 'wsi'
            'output_dir' : '',
            'patch_input_shape'  : None, 
            'patch_output_shape' : None,
        }        

    run_args = copy.deepcopy(default_run_args)
    for k, v in args.items():
        k = k[2:] # exclude the append `--`
        if k[2:] in run_args:
            run_args[k[2:]] = v
        else:
            raise Exception('Unknown CLI arg `%s`' % k)
    for k, v in run_args.items():
        if v == None:
            raise Exception('Must supply value for `--%s`.' % k)

    if args['--run-mode'] == 'tile':
        from infer.tile import InferManager
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    else:
        from infer.wsi import InferManager
        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)

