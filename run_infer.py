"""run_infer.py
Usage:
  run_infer.py [--gpu=<id>] [--model_name=<name>] [--run_mode=<mode>] [--nr_types=<n>] [--model_path=<path>] \
               [--nr_inference_workers=<n>] [--nr_post_proc_workers=<n>] [--batch_size=<n>] \
               [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] [--wsi_proc_mag=<n>] \
               [--cache_path=<path>] [--input_dir=<path>] [--input_msk_dir=<path>] \
               [--output_dir=<path>] [--patch_input_shape=<n>] [--patch_output_shape=<n>]
  run_infer.py (-h | --help)
  run_infer.py --version
Options:
  -h --help                   Show this string.
  --version                   Show version.
  --gpu=<id>                  GPU list. [default: 0]
  --model_name=<name>         Method name. [default: hovernet]
  --run_mode=<mode>           Inference mode. 'tile' or 'wsi'. [default: tile]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --model_path=<path>         Path to saved checkpoint.
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size. [default: 128]
  --ambiguous_size=<n>        Ambiguous size. [default: 128]
  --chunk_shape=<n>           Shape of chunk for processing. [default: 10000]
  --tile_shape=<n>            Shape of tiles for processing. [default: 2048]
  --wsi_proc_mag=<n>          Magnification level used for WSI processing. [default: -1]
  --cache_path=<path>         Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
  --input_dir=<path>          Path to input data directory. Assumes the files are not nested within directory.
  --input_msk_dir=<path>      Path to directory containing tissue masks. Should have the same name as corresponding WSIs. [default: '']
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

    # model_code = 'data=[pseudo_v0.5+kumar_train]'
    # model_code = 'pannuke_src'
    # model_code = 'pannuke_full_v0'
    # model_code = 'pannuke_full_v0_merge_epi'
    args['--gpu'] = '0,1,2,3'
    args['--nr_types'] = 5
    args['--model_name'] = 'hovernet'
    # args['--model_path'] = '../pretrained/pecan-hover-net-pytorch.tar'
    # args['--model_path'] = 'exp_output/models/hovernet/%s/01/net_epoch=50.tar' % model_code
    # args['--model_path'] = 'exp_output/models/hovernet/%s/01/net_epoch=50.tar' % model_code

    # args['--run_mode'] = 'wsi'
    # args['--cache_path'] = '/home/tialab-dang/cache/'
    # args['--input_dir']     = '../../dataset/RMT_mIHC_HE_Correlation/v1_scan/HE_Scans/'
    # args['--input_msk_dir'] = '../../dataset/RMT_mIHC_HE_Correlation/v1_scan/HE_masks/'
    # args['--output_dir'] = 'exp_output/prediction/RMT/%s/base/' % model_code

    model_code = 'continual_data=[v0.0]_run=[v0.0]'
    args['--run_mode'] = 'tile'
    args['--model_path'] = 'exp_output/models/hovernet/%s/net_best=[valid-tp_dice_2].tar' % model_code
    # args['--model_path'] = 'exp_output/models/hovernet/%s/net_epoch=120.tar' % model_code
    args['--cache_path'] = '/home/tialab-dang/cache/'
    args['--input_dir']     = 'dataset/rmt/sample/imgs/'
    args['--output_dir'] = 'dataset/rmt/continual_v0.0/pred/%s/' % model_code

    args['--patch_input_shape'] = 256
    args['--patch_output_shape'] = 164

    if args['--gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['--gpu']

    # raise exceptions for invalid / missing arguments
    if args['--run_mode'] != 'tile' and args['--run_mode'] != 'wsi':
        raise Exception('Mode not recognised. Use either "tile" or "wsi"')

    # TODO: exposed model kwargs ?
    if args['--model_path'] == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')
    if args['--model_name'] == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')
    nr_types = int(args['--nr_types']) if int(args['--nr_types']) > 0 else None

    method_args = {
        'method' : {
            'model_name'       : args['--model_name'],
            'model_args' : {
                'nr_types'   : nr_types,
                'mode'       : 'pannuke',
            },
            'model_path' : args['--model_path'],
        },
    }

    run_args = {
            'nr_inference_workers' : int(args['--nr_inference_workers']),
            'nr_post_proc_workers' : int(args['--nr_post_proc_workers']),
            'batch_size' : int(args['--batch_size']),
            'input_dir'  : args['--input_dir'],
            'output_dir' : args['--output_dir'],
            'patch_input_shape': int(args['--patch_input_shape']),
            'patch_output_shape': int(args['--patch_output_shape']),
        }

    if args['--run_mode'] == 'wsi':
        wsi_run_args = {
            'input_msk_dir' : args['--input_msk_dir'],
            'ambiguous_size': int(args['--ambiguous_size']),
            'chunk_shape': int(args['--chunk_shape']),
            'tile_shape': int(args['--tile_shape']),
            'cache_path' : args['--cache_path'],
            'wsi_proc_mag' : int(args['--wsi_proc_mag']),
        }
        run_args.update(wsi_run_args) 

    if args['--run_mode'] == 'tile':
        from infer.tile import InferManager
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    else:
        from infer.wsi import InferManager
        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)