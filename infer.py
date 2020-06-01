
"""infer.py

Usage:
  infer.py [--gpu=<id>] [--mode=<mode>] [--model=<path>] [--batch_size=<n>] [--num_workers=<n>] [--input_dir=<path>] [--output_dir=<path>] [--tile_size=<n>] [--return_masks]
  infer.py (-h | --help)
  infer.py --version

Options:
  -h --help            Show this string.
  --version            Show version.
  --gpu=<id>           GPU list. [default: 0]
  --mode=<mode>        Inference mode. 'roi' or 'wsi'. [default: roi]
  --model=<path>       Path to saved checkpoint.
  --input_dir=<path>   Directory containing input images/WSIs.
  --output_dir=<path>  Directory where the output will be saved. [default: output/]
  --batch_size=<n>     Batch size. [default: 25]
  --num_workers=<n>    Number of workers. [default: 12]
  --tile_size=<n>      Size of tiles (assumes square shape). [default: 20000]
  --return_masks       Whether to return cropped nuclei masks
"""


import argparse
import glob
import importlib
import json
import math
import os
import sys
import re
import random
import warnings
import time

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from docopt import docopt
import tqdm
from progress.bar import Bar as ProgressBar

from dataloader.infer_loader import SerializeFile, SerializeWSI

from config import Config
from misc.utils import rm_n_mkdir
from misc.viz_utils import visualize_instances
import postproc.process_utils as proc_utils
import dataset



####
class InferTile(Config):
    """
    Run inference on tiles
    """
    def __init__(self):
        super().__init__()
        self.input_shape = self.shape_info['test']['input_shape']
        self.mask_shape = self.shape_info['test']['mask_shape']
        dataset_info = getattr(dataset, self.dataset_name)(self.type_classification)
        self.nr_types = dataset_info.nr_types
        self.colour_dict = dataset_info.class_colour
    
    ####
    def load_args(self, args):
        """
        Load arguments
        """
        # Paths
        self.model_path = args['--model']
        self.input_dir = args['--input_dir']
        self.output_dir = args['--output_dir']

        # Processing
        self.batch_size = int(args['--batch_size'])
        self.num_workers = int(args['--num_workers'])

    ####
    def load_model(self):
        """
        Create the model, load the checkpoint and define
        associated run steps to process each data batch
        """
        netdesc = importlib.import_module('model.net_desc')
        net = netdesc.HoVerNet(3, self.nr_types)

        saved_state_dict = torch.load(self.model_path)
        net.load_state_dict(saved_state_dict, strict=False)
        net = torch.nn.DataParallel(net).to('cuda')

        run_step = importlib.import_module('model.run_desc')
        run_step = getattr(run_step, 'infer_step') 
        def infer_step(input_batch): 
            return run_step(input_batch, net)

        return infer_step

    ####
    def __process_single_tile(self, filename, img_shape):
        """
        Process a single image tile < 5000x5000 in size.
        """
        run_step = self.load_model()

        # * depend on the number of samples and their size, this may be less efficient
        dataset = SerializeFile(filename, self.input_shape, self.mask_shape)
        dataloader = data.DataLoader(dataset,
                                    num_workers=self.num_workers,
                                    batch_size=self.batch_size,
                                    drop_last=False)

        patch_accmulator = []
        for batch_idx, batch_data in enumerate(dataloader):
            sample_list, sample_info_list = batch_data
            sample_output = run_step(sample_list)['raw']

            # combine the output from each branch
            batch_list = []
            for output_keys in sample_output.keys():
                batch_output_tmp = sample_output[output_keys]
                if len(batch_output_tmp.shape) == 3:
                    batch_output_tmp = np.expand_dims(batch_output_tmp, -1) 
                batch_list.append(batch_output_tmp)
            sample_output = np.concatenate(batch_list, axis=-1)

            # split at the batch dimension
            sample_output = np.split(sample_output, sample_output.shape[0], axis=0)

            sample_info_list = sample_info_list.numpy()
            sample_output = list(zip(sample_output, sample_info_list))
            patch_accmulator.extend(sample_output)

        # re-assemble the prediction 
        patch_accmulator = sorted(patch_accmulator, key=lambda x: [x[1][0], x[1][1]]) # sort by where the patch appears in the original image
        patches, patch_info = zip(*patch_accmulator)

        patch_shape = np.squeeze(patches[0]).shape
        ch = 1 if len(patch_shape) == 2 else patch_shape[-1]

        patch_info = np.array(patch_info)
        nr_row = patch_info[:, 0].max() + 1
        nr_col = patch_info[:, 1].max() + 1
        pred_map = np.concatenate(patches, axis=0)
        pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 \
            else np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (patch_shape[0] * nr_row,
                                            patch_shape[1] * nr_col, ch))
        pred_map = np.squeeze(pred_map[:img_shape[0], :img_shape[1]])
        
        return pred_map

    ####
    def process_all_files(self):
        """
        Process image files within a directory.
        For each image, the function will:
        1) Load the image
        2) Extract patches and pass through the network
        3) Return output .mat file and overlay
        """
        save_dir = self.output_dir
        file_list = glob.glob('%s/*' % self.input_dir)
        file_list.sort()  # ensure same order

        rm_n_mkdir(save_dir)
        for idx, filename in enumerate(file_list):
            basename = os.path.basename(filename)
            basename = os.path.splitext(basename)[0]
            
            sys.stdout.write("\rProcessing %s (%d/%d)" % (
                basename, idx + 1, len(file_list)))
            sys.stdout.flush()

            ###
            # load original image - used for overlay
            img = cv2.imread(filename) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_shape = img.shape

            ###
            # extract patches and run inference
            pred_map = self.__process_single_tile(filename, img_shape)
            # apply post-processing
            pred_inst, pred_type = proc_utils.process_instance(pred_map, self.type_classification, nr_types=6)
            
            # generate overlay
            overlaid_output = visualize_instances(img, pred_inst, colours=self.colour_dict)
            overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)

            cv2.imwrite('%s/%s.png' % (self.output_dir, basename),
                        overlaid_output)

            if self.type_classification:
                sio.savemat('%s/%s.mat' % (self.output_dir, basename),
                            {'inst_map': pred_inst,
                             'type_map': pred_type})
            else:
                sio.savemat('%s/%s.mat' % (self.output_dir, basename),
                            {'inst_map': pred_inst})
                
    
####
class InferWSI(Config):
    """
    Run inference on WSIs
    """
    def __init__(self):
        super().__init__()
        self.input_shape = self.shape_info['test']['input_shape']
        self.mask_shape = self.shape_info['test']['mask_shape']
        self.proc_lvl = 0  # WSI level at which to process
        self.tiss_seg = True  # only process tissue areas
        self.tiss_lvl = 3  # WSI level at which perform tissue segmentation
        dataset_info = getattr(dataset, self.dataset_name)(self.type_classification)
        self.nr_types = dataset_info.nr_types
        self.colour_dict = dataset_info.class_colour

    ####
    def load_args(self, args):
        """
        Load arguments
        """
        # Tile Size
        self.tile_size = int(args['--tile_size'])
        # Paths
        self.model_path = args['--model']
        self.input_dir = args['--input_dir']
        self.output_dir = args['--output_dir']

        # Processing
        self.batch_size = int(args['--batch_size'])
        self.num_workers = int(args['--num_workers'])
        # Below specific to WSI processing
        self.return_masks = args['--return_masks']

    ####
    def load_model(self):
        """
        Create the model, then load the checkpoint and define
        associated run steps to process each data batch
        """
        netdesc = importlib.import_module('model.net_desc')
        net = netdesc.HoVerNet(3, self.nr_types)

        saved_state_dict = torch.load(self.model_path)
        net.load_state_dict(saved_state_dict, strict=False)
        net = torch.nn.DataParallel(net).to('cuda')

        run_step = importlib.import_module('model.run_desc')
        run_step = getattr(run_step, 'infer_step')

        def infer_step(input_batch):
            return run_step(input_batch, net)

        return infer_step
    
    ####
    def load_wsi(self, file_path, wsi_ext):
        """
        Load WSI using OpenSlide. Note, if using JP2, appropriate
        matlab scripts need to be placed in the working directory
        Args:
            wsi_ext: file extension of the whole-slide image
        """

        if wsi_ext == 'jp2':
            # are not saved as an image pyramid- instead uses wavelet compression.
            # save information as if it was an image pyramid
            self.wsi_open = glymur.Jp2k(file_path)
            fullres = self.wsi_open[:] #! may be computationally expensive - check!
            shape_x = fullres.shape[1]
            shape_y = fullres.shape[0]
            self.level_downsamples = [1, 2, 4, 8, 16, 32, 64]
            self.level_count = len(level_downsamples)
            self.level_dimensions = []
            for i in range(self.level_count):
                self.level_dimensions.append((int(round(shape_x / self.level_downsamples[i])), int(round(shape_x / self.level_downsamples[i]))))
            self.scan_resolution = [0.275, 0.275]  # scan resolution of the Omnyx scanner
        else:
            self.wsi_open = ops.OpenSlide(file_path)
            self.level_downsamples = self.wsi_open.level_downsamples
            self.level_count = self.wsi_open.level_count
            self.level_dimensions = []
            # flipping cols into rows (Openslide to python format)
            for i in range(self.level_count):
                self.level_dimensions.append([self.wsi_open.level_dimensions[i][1], self.wsi_open.level_dimensions[i][0]])
            self.scan_resolution = [float(self.wsi_open.properties.get('openslide.mpp-x')),
                               float(self.wsi_open.properties.get('openslide.mpp-y'))]
        
    ####
    def read_region(self, location, level, patch_size, wsi_ext):
        """
        Loads a region from a WSI object
        
        Args:
            location: top left coordinates of patch
            level: level of WSI pyramid at which to extract
            patch_size: patch size to extract
            wsi_ext: WSI file extension
        
        Returns:
            patch: extracted patch (np array)
        """
        if wsi_ext == 'jp2':
            patch = self.wsi_open[location[1]:location[1] + patch_size[1]:self.ds_factor, 
                                  location[0]:location[0] + patch_size[0]:self.ds_factor, :]
        else:
            patch = self.wsi_open.read_region(location, level, patch_size)
            r, g, b, _ = cv2.split(np.array(patch))
            patch = cv2.merge([r, g, b])
        return patch
    
    ####
    def tile_coords(self):
        """
        Get the tile coordinates and dimensions for processing at level 0
        """

        self.im_w = self.level_dimensions[self.proc_lvl][1]
        self.im_h = self.level_dimensions[self.proc_lvl][0]

        self.nr_tiles_h = math.ceil(self.im_h / self.tile_size)
        self.nr_tiles_w = math.ceil(self.im_w / self.tile_size)

        step_h = self.tile_size
        step_w = self.tile_size

        self.tile_info = []

        for row in range(self.nr_tiles_h):
            for col in range(self.nr_tiles_w):
                start_h = row * step_h
                start_w = col * step_w
                if row == self.nr_tiles_h - 1:
                    extra_h = self.im_h - (self.nr_tiles_h * step_h)
                    dim_h = step_h + extra_h
                else:
                    dim_h = step_h
                if col == self.nr_tiles_w - 1:
                    extra_w = self.im_w - (self.nr_tiles_w * step_w)
                    dim_w = step_w + extra_w
                else:
                    dim_w = step_w
                self.tile_info.append(
                    (int(start_w), int(start_h), int(dim_w), int(dim_h)))

    ####
    def load_filenames(self):
        """
        Get the list of all files to process
        """
        search_path = '%s/*' % self.input_dir
        search_path = re.sub('([\[\]])', '[\\1]', search_path)
        file_list = glob.glob(search_path)
        file_list.sort()  # ensure same order

        return file_list

    ####
    def __process_single_tile(self, dataloader, tile_idx, tile_total):
        """
        run inference on a single tile
        """
        run_step = self.load_model()

        pbar_format = 'Processing tile (%d/%d) of %s:' % (tile_idx+1, tile_total, self.basename) + \
                      '|{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm.tqdm(total=len(dataloader), leave=True, bar_format=pbar_format)

        patch_accmulator = []
        for batch_idx, batch_data in enumerate(dataloader):
            sample_list, sample_info_list = batch_data
            sample_output = run_step(sample_list)['raw']

            # combine the output from each branch
            batch_list = []
            for output_keys in sample_output.keys():
                batch_output_tmp = sample_output[output_keys]
                if len(batch_output_tmp.shape) == 3:
                    batch_output_tmp = np.expand_dims(batch_output_tmp, -1)
                batch_list.append(batch_output_tmp)
            sample_output = np.concatenate(batch_list, axis=-1)

            sample_output = np.split(sample_output, sample_output.shape[0], axis=0)

            sample_info_list = sample_info_list.numpy()
            sample_output = list(zip(sample_output, sample_info_list))
            patch_accmulator.extend(sample_output)
            pbar.update()
        pbar.close()  # to flush out the bar in case a new bar is needed

        # re-assemble the accumulated prediction
        # TODO: Synchronize or enclose within the loader parsing?
        patch_accmulator = sorted(patch_accmulator, key=lambda x: [x[1][0], x[1][1]])
        patches, patch_info = zip(*patch_accmulator)

        patch_shape = np.squeeze(patches[0]).shape
        ch = 1 if len(patch_shape) == 2 else patch_shape[-1]

        patch_info = np.array(patch_info)
        nr_row = patch_info[:, 0].max() + 1
        nr_col = patch_info[:, 1].max() + 1
        pred_map = np.concatenate(patches, axis=0)
        pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 \
            else np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (patch_shape[0] * nr_row,
                                         patch_shape[1] * nr_col, ch))
        return pred_map

    ####
    def __process_single_wsi(self, file_path):
        """
        Process a single WSI and save the results

        Args:
            file_path: path of the WSI file to process
        """
        # Load the WSI
        wsi_ext = file_path.split('.')[-1]
        self.load_wsi(file_path, wsi_ext)

        self.ds_factor = self.level_downsamples[self.proc_lvl]

        # get the level for tissue mask generation
        is_valid_tissue_level = True
        tissue_level = self.tiss_lvl
        if len(self.level_downsamples) > 1:
            tissue_level = len(self.level_downsamples) - 1  # to avoid tissue segmentation at level 0
        else:
            is_valid_tissue_level = False

        if self.tiss_seg & is_valid_tissue_level:
            # read WSI at low-resolution
            ds_img = self.read_region(
                (0, 0),
                tissue_level,
                (self.level_dimensions[tissue_level][1], self.level_dimensions[tissue_level][0]),
                wsi_ext
            )
            proc_scale = 1 / np.ceil(np.max(ds_img.shape) / 5000) # used to reduce processing time
            tissue = proc_utils.get_tissue_mask(ds_img, proc_scale) # generate the tissue mask
        else:
            self.tiss_seg = False 

        # Coordinate info for tile processing
        self.tile_coords()
        # offset 
        self.offset = (self.input_shape[0] - self.mask_shape[0]) / 2

        mask_list_all = []
        type_list_all = []
        cent_list_all = []
        # Run inference tile by tile - if self.tiss_seg == True, only process tissue regions
        for tile_idx, tile_info in enumerate(self.tile_info):
            
            dataset = SerializeWSI(file_path, wsi_ext, tile_info, self.input_shape, self.mask_shape, self.ds_factor, self.proc_lvl, tissue)
            dataloader = data.DataLoader(dataset,
                                     num_workers=self.num_workers,
                                     batch_size=self.batch_size,
                                     drop_last=False)

            tile_coords = (tile_info[0], tile_info[1]) 

            # run inference on a single tile
            pred_map = self.__process_single_tile(dataloader, tile_idx, len(self.tile_info))

            # perform post-processing
            mask_list, type_list, cent_list = proc_utils.process_instance_wsi(
                pred_map, self.nr_types, tile_coords, self.return_masks, offset=self.offset)

            # add tile predictions to overall prediction list
            mask_list_all.extend(mask_list)
            type_list_all.extend(type_list)
            cent_list_all.extend(cent_list)

        np.savez('%s/%s/%s.npz' % (
            self.output_dir, self.basename, self.basename),
            mask=mask_list_all, type=type_list_all, centroid=cent_list_all
        )
    
    ####
    def process_all_files(self):
        """
        Process each file one at a time
        """
        if os.path.isdir(self.output_dir) == False:
            rm_n_mkdir(self.output_dir)
        
        self.file_list = self.load_filenames()

        for filename in self.file_list:
            basename = os.path.basename(filename)
            self.basename = os.path.splitext(basename)[0]
            # this will overwrite file is it was processed previously
            rm_n_mkdir(self.output_dir + '/' + self.basename)
            start_time_total = time.time()
            self.__process_single_wsi(filename)

            end_time_total = time.time()
            print('. FINISHED. Time: ', time_it(start_time_total, end_time_total), 'secs')



#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = docopt(__doc__, version='HoVer-Net Pytorch Inference v1.0')
    print(args)

    args['--model'] = 'logs/tmp/00/net_epoch=50.tar'
    args['--input_dir'] =  '/home/simon/Desktop/Projects/Nuclei_seg/hovernet_inference/WSIs'
    args['--output_dir'] = 'output/'
    args['--mode'] = 'wsi'
    args['--gpu'] = '0'
    args['--batch_size'] = '8'

    os.environ['CUDA_VISIBLE_DEVICES'] = args['--gpu']

    # raise exceptions for invalid / missing arguments
    if args['--model'] == None:
        raise Exception('A model path must be supplied as an argument with --model.')
    if args['--mode'] != 'tile' and args['--mode'] != 'wsi':
        raise Exception('Mode not recognised. Use either "tile" or "wsi"')
    if args['--input_dir'] == None:
        raise Exception('An input directory must be supplied as an argument with --input_dir.')
    if args['--input_dir'] == args['--output_dir']:
        raise Exception('Input and output directories should not be the same- otherwise input directory will be overwritten.')

    # import libraries for WSI processing
    if args['--mode'] == 'wsi':
        import openslide as ops
    
    if args['--mode'] == 'tile':
        infer = InferTile()
    elif args['--mode'] == 'wsi':  # currently saves results per tile
        infer = InferWSI()
        
    infer.load_args(args)
    infer.process_all_files()
        
