
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

import time
from multiprocessing import Pool, Lock
import multiprocessing as mp
mp.set_start_method('spawn', True) # ! must be at top for VScode debugging
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

from misc.utils import rm_n_mkdir, cropping_center, get_bounding_box
from postproc import hover

from . import base
import openslide

thread_lock = Lock()
####
def _init_worker_child(lock_):
    global lock
    lock = lock_
####
def _remove_inst(inst_map, remove_id_list):
    for inst_id in remove_id_list:
        inst_map[inst_map == inst_id] = 0
    return inst_map
####
def _get_patch_top_left_info(img_shape, input_size, output_size):
    in_out_diff = input_size - output_size
    nr_step = np.floor((img_shape - in_out_diff) / output_size) + 1
    last_output_coord = (in_out_diff // 2) + (nr_step) * output_size
    # generating subpatches index from orginal
    output_tl_y_list = np.arange(in_out_diff[0] // 2, last_output_coord[0], output_size[0], dtype=np.int32)
    output_tl_x_list = np.arange(in_out_diff[1] // 2, last_output_coord[1], output_size[1], dtype=np.int32)
    output_tl_y_list, output_tl_x_list = np.meshgrid(output_tl_y_list, output_tl_x_list)
    output_tl = np.stack([output_tl_y_list.flatten(), output_tl_x_list.flatten()], axis=-1)
    input_tl = output_tl - in_out_diff // 2
    return input_tl, output_tl
#### all must be np.array
def _get_tile_info(img_shape, tile_shape, ambiguous_size=128):
    # * get normal tiling set
    tile_grid_top_left, _ = _get_patch_top_left_info(img_shape, tile_shape, tile_shape)
    tile_grid_bot_right = []
    for idx in list(range(tile_grid_top_left.shape[0])):
        tile_tl = tile_grid_top_left[idx][:2]
        tile_br = tile_tl + tile_shape
        axis_sel = tile_br > img_shape
        tile_br[axis_sel] = img_shape[axis_sel]
        tile_grid_bot_right.append(tile_br)
    tile_grid_bot_right = np.array(tile_grid_bot_right)
    tile_grid = np.stack([tile_grid_top_left, tile_grid_bot_right], axis=1)
    tile_grid_x = np.unique(tile_grid_top_left[:,1])
    tile_grid_y = np.unique(tile_grid_top_left[:,0])
    # * get tiling set to fix vertical and horizontal boundary between tiles
    # for sanity, expand at boundary `ambiguous_size` to both side vertical and horizontal
    stack_coord = lambda x: np.stack([x[0].flatten(), x[1].flatten()], axis=-1)
    tile_boundary_x_top_left  = np.meshgrid(tile_grid_y, tile_grid_x[1:] - ambiguous_size)
    tile_boundary_x_bot_right = np.meshgrid(tile_grid_y + tile_shape[0], tile_grid_x[1:] + ambiguous_size)
    tile_boundary_x_top_left  = stack_coord(tile_boundary_x_top_left)
    tile_boundary_x_bot_right = stack_coord(tile_boundary_x_bot_right)
    tile_boundary_x = np.stack([tile_boundary_x_top_left, tile_boundary_x_bot_right], axis=1)
    #
    tile_boundary_y_top_left  = np.meshgrid(tile_grid_y[1:] - ambiguous_size, tile_grid_x)
    tile_boundary_y_bot_right = np.meshgrid(tile_grid_y[1:] + ambiguous_size, tile_grid_x+tile_shape[1])
    tile_boundary_y_top_left  = stack_coord(tile_boundary_y_top_left)
    tile_boundary_y_bot_right = stack_coord(tile_boundary_y_bot_right)
    tile_boundary_y = np.stack([tile_boundary_y_top_left, tile_boundary_y_bot_right], axis=1)
    tile_boundary = np.concatenate([tile_boundary_x, tile_boundary_y], axis=0)
    # * get tiling set to fix the intersection of 4 tiles
    tile_cross_top_left  = np.meshgrid(tile_grid_y[1:] -  2 * ambiguous_size, tile_grid_x[1:] - 2 * ambiguous_size)
    tile_cross_bot_right = np.meshgrid(tile_grid_y[1:] +  2 * ambiguous_size, tile_grid_x[1:] + 2 * ambiguous_size)
    tile_cross_top_left  = stack_coord(tile_cross_top_left)
    tile_cross_bot_right = stack_coord(tile_cross_bot_right)
    tile_cross = np.stack([tile_cross_top_left, tile_cross_bot_right], axis=1)
    return tile_grid, tile_boundary, tile_cross
### 
def _get_chunk_patch_info(img_shape, chunk_input_shape, patch_input_shape, patch_output_shape):
    round_to_multiple = lambda x, y: np.floor(x / y) * y
    patch_diff_shape = patch_input_shape - patch_output_shape

    chunk_output_shape = chunk_input_shape - patch_diff_shape
    chunk_output_shape = round_to_multiple(chunk_output_shape, patch_output_shape).astype(np.int64)
    chunk_input_shape  = (chunk_output_shape + patch_diff_shape).astype(np.int64)

    patch_input_tl_list, _ = _get_patch_top_left_info(img_shape, patch_input_shape, patch_output_shape)
    patch_input_br_list = patch_input_tl_list + patch_input_shape
    patch_output_tl_list = patch_input_tl_list + patch_diff_shape
    patch_output_br_list = patch_output_tl_list + patch_output_shape 
    patch_info_list = np.stack(
                        [np.stack([patch_input_tl_list, patch_input_br_list], axis=1),
                         np.stack([patch_output_tl_list, patch_output_br_list], axis=1)], axis=1)

    chunk_input_tl_list, _ = _get_patch_top_left_info(img_shape, chunk_input_shape, chunk_output_shape)
    chunk_input_br_list = chunk_input_tl_list + chunk_input_shape
    # * correct the coord so it stay within source image
    y_sel = np.nonzero(chunk_input_br_list[:,0] > img_shape[0])[0]
    x_sel = np.nonzero(chunk_input_br_list[:,1] > img_shape[1])[0]
    chunk_input_br_list[y_sel, 0] = (img_shape[0] - patch_diff_shape[0]) - chunk_input_tl_list[y_sel,0] 
    chunk_input_br_list[x_sel, 1] = (img_shape[1] - patch_diff_shape[1]) - chunk_input_tl_list[x_sel,1] 
    chunk_input_br_list[y_sel, 0] = round_to_multiple(chunk_input_br_list[y_sel, 0], patch_output_shape[0]) 
    chunk_input_br_list[x_sel, 1] = round_to_multiple(chunk_input_br_list[x_sel, 1], patch_output_shape[1]) 
    chunk_input_br_list[y_sel, 0] += chunk_input_tl_list[y_sel, 0] + patch_diff_shape[0]
    chunk_input_br_list[x_sel, 1] += chunk_input_tl_list[x_sel, 1] + patch_diff_shape[1]
    chunk_output_tl_list = chunk_input_tl_list + patch_diff_shape // 2
    chunk_output_br_list = chunk_input_br_list - patch_diff_shape // 2 # may off pixels
    chunk_info_list = np.stack(
                        [np.stack([chunk_input_tl_list , chunk_input_br_list], axis=1),
                         np.stack([chunk_output_tl_list, chunk_output_br_list], axis=1)], axis=1)

    return chunk_info_list, patch_info_list
####
def _post_proc_para_wrapper(pred_map_mmap_path, tile_info, func, func_kwargs):
    idx, tile_tl, tile_br = tile_info
    wsi_pred_map_ptr = np.load(pred_map_mmap_path, mmap_mode='r')
    tile_pred_map = wsi_pred_map_ptr[tile_tl[0] : tile_br[0],
                                     tile_tl[1] : tile_br[1]]
    tile_pred_map = np.array(tile_pred_map) # from mmap to ram
    return func(tile_pred_map, **func_kwargs), tile_info
####
def _assemble_and_flush(wsi_pred_map_mmap_path, chunk_info, patch_output_list):
    # write to newly created holder for this wsi
    wsi_pred_map_ptr = np.load(wsi_pred_map_mmap_path, mmap_mode='r+')
    chunk_pred_map = wsi_pred_map_ptr[chunk_info[1][0][0]: chunk_info[1][1][0],
                                      chunk_info[1][0][1]: chunk_info[1][1][1]] 
    if patch_output_list is None:
        chunk_pred_map[:] = 0 # zero flush when there is no-results
        print(chunk_info.flatten(), 'flush 0')
        return

    for pinfo in patch_output_list:
        pcoord, pdata = pinfo
        pdata = np.squeeze(pdata)
        pcoord = np.squeeze(pcoord)[:2]
        chunk_pred_map[pcoord[0] : pcoord[0] + pdata.shape[0],
                       pcoord[1] : pcoord[1] + pdata.shape[1]] = pdata
    print(chunk_info.flatten(), 'pass')
    return
####
class Inferer(base.Inferer):

    def __run_model(self, patch_top_left_list):
        # TODO: the cost of creating dataloader may not be cheap ?
        dataset = SerializeArray('%s/cache_chunk.npy' % self.wsi_cache_path, 
                                patch_top_left_list, self.patch_input_shape)

        dataloader = data.DataLoader(dataset,
                            num_workers=self.nr_inference_workers,
                            batch_size=self.batch_size,
                            drop_last=False)

        pbar = tqdm.tqdm(desc='Process Patches', leave=True,
                    total=int(len(dataloader)), 
                    ncols=80, ascii=True, position=0)

        accumulated_patch_output = []
        for batch_idx, batch_data in enumerate(dataloader):
            sample_data_list, sample_info_list = batch_data
            sample_output_list = self.run_step(sample_data_list)
            sample_info_list = sample_info_list.numpy()
            curr_batch_size = sample_output_list.shape[0]
            sample_output_list = np.split(sample_output_list, curr_batch_size, axis=0) 
            sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)
            sample_output_list = list(zip(sample_info_list, sample_output_list))
            accumulated_patch_output.extend(sample_output_list)
            pbar.update()
        pbar.close()
        return accumulated_patch_output

    def __select_valid_patches(self, patch_info_list):
        down_sample_ratio = self.wsi_mask.shape[0] / self.wsi_proc_shape[0]
        for _ in range(len(patch_info_list)):
            patch_info = patch_info_list.pop(0)
            patch_info = np.squeeze(patch_info)
            # get the box at corresponding mag of the mask
            output_bbox = patch_info[1] * down_sample_ratio
            output_bbox = np.rint(output_bbox).astype(np.int64)
            # coord of the output of the patch (i.e center regions)
            output_roi = self.wsi_mask[output_bbox[0][0]:output_bbox[1][0],
                                       output_bbox[0][1]:output_bbox[1][1]]
            if np.sum(output_roi) > 0:
                patch_info_list.append(patch_info)
        return patch_info_list

    def __get_raw_prediction(self, chunk_info_list, patch_info_list):

        # 1 dedicated thread just to write results back to disk
        proc_pool = Pool(processes=1, 
                        initializer=_init_worker_child, 
                        initargs=(thread_lock,))
        wsi_pred_map_mmap_path = '%s/pred_map.npy' % self.wsi_cache_path

        masking = lambda x, a, b: (a <= x) & (x <= b)
        for chunk_info in chunk_info_list:
            # select patch basing on top left coordinate of input
            start_coord = chunk_info[0,0]
            end_coord = chunk_info[0,1] - self.patch_input_shape
            selection = masking(patch_info_list[:,0,0,0], start_coord[0], end_coord[0]) \
                      & masking(patch_info_list[:,0,0,1], start_coord[1], end_coord[1])
            chunk_patch_info_list = np.array(patch_info_list[selection]) # * do we need copy ?
            chunk_patch_info_list = np.split(chunk_patch_info_list, chunk_patch_info_list.shape[0], axis=0)

            # further select only the patches within the provided mask
            chunk_patch_info_list = self.__select_valid_patches(chunk_patch_info_list)

            # there no valid patches, so flush 0 and skip
            if len(chunk_patch_info_list) == 0:
                proc_pool.apply_async(_assemble_and_flush, 
                                    args=(wsi_pred_map_mmap_path, chunk_info, None))
                continue

            # shift the coordinare from wrt slide to wrt chunk
            chunk_patch_info_list = np.array(chunk_patch_info_list)
            chunk_patch_info_list -= chunk_info[:,0]
            chunk_data = self.wsi_handler.read_region(chunk_info[0][0][::-1], self.wsi_proc_mag, 
                                                     (chunk_info[0][1] - chunk_info[0][0])[::-1])
            chunk_data = np.array(chunk_data)[...,:3]
            np.save('%s/cache_chunk.npy' % self.wsi_cache_path, chunk_data)

            patch_output_list = self.__run_model(chunk_patch_info_list[:,0,0])

            proc_pool.apply_async(_assemble_and_flush, 
                                    args=(wsi_pred_map_mmap_path, 
                                        chunk_info, patch_output_list))
   
        proc_pool.close()
        proc_pool.join()
        return

    def __dispatch_post_processing(self, tile_info_list, callback):

        if self.nr_procs > 0: 
            proc_pool = Pool(processes=self.nr_post_proc_workers, 
                            initializer=_init_worker_child, 
                            initargs=(thread_lock,))

        wsi_pred_map_mmap_path = '%s/pred_map.npy' % self.wsi_cache_path
        for idx in list(range(tile_info_list.shape[0])):
            tile_tl = tile_info_list[idx][0]
            tile_br = tile_info_list[idx][1]
            tile_info = (idx, tile_tl, tile_br)
            func_kwargs = {
                'nr_types' : None,
                'return_centroids' : True
            }

            # mp.Array()
            # TODO: standarize protocol
            if self.nr_procs > 0:
                proc_pool.apply_async(_post_proc_para_wrapper, callback=callback, 
                                    args=(wsi_pred_map_mmap_path, tile_info, 
                                        hover.process, func_kwargs))
            else:
                results = _post_proc_para_wrapper(wsi_pred_map_mmap_path, tile_info, 
                                        hover.process, func_kwargs)
                callback(results)
        if self.nr_procs > 0:
            proc_pool.close()
            proc_pool.join()
        return

    def _parse_args(self, run_args):
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        return

    def process_single_file(self, wsi_path, msk_path, output_path):
        # TODO: customize universal file handler to sync the protocol
        ambiguous_size = self.ambiguous_size
        tile_shape = (np.array(self.tile_shape)).astype(np.int64)
        chunk_input_shape  = np.array(self.chunk_shape)
        patch_input_shape  = np.array(self.patch_input_shape)
        patch_output_shape = np.array(self.patch_output_shape)

        self.wsi_handler = openslide.OpenSlide(wsi_path)
        # TODO: customize read lv
        self.wsi_proc_mag   = 0 # w.r.t source magnification
        self.wsi_proc_shape = self.wsi_handler.level_dimensions[self.wsi_proc_mag] # TODO: turn into func
        self.wsi_proc_shape = np.array(self.wsi_proc_shape[::-1]) # to Y, X

        # TODO: simplify / protocolize this
        self.wsi_mask = cv2.imread(msk_path)
        self.wsi_mask = cv2.cvtColor(self.wsi_mask, cv2.COLOR_BGR2GRAY)
        self.wsi_mask[self.wsi_mask > 0] = 1

        # * declare holder for output
        # create a memory-mapped .npy file with the predefined dimensions and dtype
        out_ch = 3 # TODO: dynamicalize this, retrieve from model?
        self.wsi_inst_info  = {} 
        self.wsi_inst_map   = np.zeros(self.wsi_proc_shape, dtype=np.int32)
        # warning, the value within this is uninitialized
        self.wsi_pred_map = np.lib.format.open_memmap(
                                            '%s/pred_map.npy' % self.wsi_cache_path, mode='w+',
                                            shape=tuple(self.wsi_proc_shape) + (out_ch,), 
                                            dtype=np.float32)

        # * raw prediction
        start = time.perf_counter()
        chunk_info_list, patch_info_list = _get_chunk_patch_info(
                                                self.wsi_proc_shape, chunk_input_shape, 
                                                patch_input_shape, patch_output_shape)
        self.__get_raw_prediction(chunk_info_list, patch_info_list)
        end = time.perf_counter()
        print('Inference Time: ',  end - start)

        # TODO: deal with error banding
        ##### * post proc
        start = time.perf_counter()
        tile_coord_set = _get_tile_info(self.wsi_proc_shape, tile_shape, ambiguous_size)
        tile_grid_info, tile_boundary_info, tile_cross_info = tile_coord_set

        ####################### * Callback can only receive 1 arg
        def post_proc_normal_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                return # when there is nothing to do

            top_left = pos_args[1][::-1]
            with thread_lock:
                wsi_max_id = 0 

                # ! WARNING: 
                # ! inst ID may not be contiguous, 
                # ! hence must use max as safeguard

                if len(self.wsi_inst_info) > 0:
                    wsi_max_id = max(self.wsi_inst_info.keys()) 
                for inst_id, inst_info in inst_info_dict.items():
                    # now correct the coordinate wrt to wsi
                    inst_info['bbox']     += top_left
                    inst_info['contour']  += top_left
                    inst_info['centroid'] += top_left
                    self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
                pred_inst[pred_inst > 0] += wsi_max_id
                self.wsi_inst_map[tile_tl[0] : tile_br[0],
                                  tile_tl[1] : tile_br[1]] = pred_inst
            return
        ####################### * Callback can only receive 1 arg
        def post_proc_fixing_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                return # when there is nothing to do

            top_left = pos_args[1][::-1]
            with thread_lock:
                # for fixing the boundary, keep all nuclei split at boundary (i.e within unambigous region)
                # of the existing prediction map, and replace all nuclei within the region with newly predicted

                # ! WARNING: 
                # ! inst ID may not be contiguous, 
                # ! hence must use max as safeguard

                # ! must get before the removal happened
                wsi_max_id = 0 
                if len(self.wsi_inst_info) > 0:
                    wsi_max_id = max(self.wsi_inst_info.keys()) 

                # * exclude ambiguous out from old prediction map
                # check 1 pix of 4 edges to find nuclei split at boundary
                roi_inst = self.wsi_inst_map[tile_tl[0] : tile_br[0],
                                             tile_tl[1] : tile_br[1]]
                roi_inst = np.copy(roi_inst)
                roi_edge = np.concatenate([roi_inst[[0,-1],:].flatten(),
                                           roi_inst[:,[0,-1]].flatten()])
                roi_boundary_inst_list = np.unique(roi_edge)[1:] # exclude background
                roi_inner_inst_list = np.unique(roi_inst)[1:]  
                roi_inner_inst_list = np.setdiff1d(roi_inner_inst_list, 
                                                roi_boundary_inst_list, 
                                                assume_unique=True)
                roi_inst = _remove_inst(roi_inst, roi_inner_inst_list)
                self.wsi_inst_map[tile_tl[0] : tile_br[0],
                             tile_tl[1] : tile_br[1]] = roi_inst
                for inst_id in roi_inner_inst_list:
                    self.wsi_inst_info.pop(inst_id, None)

                # * exclude unambiguous out from new prediction map
                # check 1 pix of 4 edges to find nuclei split at boundary
                roi_edge = pred_inst[roi_inst > 0] # remove all overlap
                boundary_inst_list = np.unique(roi_edge) # no background to exclude                
                inner_inst_list = np.unique(pred_inst)[1:]  
                inner_inst_list = np.setdiff1d(inner_inst_list, 
                                            boundary_inst_list, 
                                            assume_unique=True)              
                pred_inst = _remove_inst(pred_inst, boundary_inst_list)

                # * proceed to overwrite
                for inst_id in inner_inst_list:
                    inst_info = inst_info_dict[inst_id]
                    # now correct the coordinate wrt to wsi
                    inst_info['bbox']     += top_left
                    inst_info['contour']  += top_left
                    inst_info['centroid'] += top_left
                    self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
                pred_inst[pred_inst > 0] += wsi_max_id
                pred_inst = roi_inst + pred_inst
                self.wsi_inst_map[tile_tl[0] : tile_br[0],
                                  tile_tl[1] : tile_br[1]] = pred_inst
            return        
        #######################
        # * must be in sequential ordering
        self.__dispatch_post_processing(tile_grid_info, post_proc_normal_tile_callback)
        self.__dispatch_post_processing(tile_boundary_info, post_proc_fixing_tile_callback)
        self.__dispatch_post_processing(tile_cross_info, post_proc_fixing_tile_callback)
        end = time.perf_counter()
        print('Post Proc Time: ', end - start)


        # import pickle
        # np.save('pred_inst.npy', self.wsi_inst_map)
        # with open("nuclei_dict.pickle", "wb") as handle:
        #     pickle.dump(self.wsi_inst_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def process_wsi_list(self, run_args):
        self._parse_args(run_args) 

        wsi_path_list = glob.glob(self.input_wsi_dir + '/*.tif')       
        for wsi_path in wsi_path_list:
            # may not work, such as when name is TCGA etc.
            wsi_base_name = os.path.basename(wsi_path).split('.')[:-1]
            msk_path = '%s/%s.png' % (self.input_msk_dir, wsi_base_name)
            self.process_single_file(wsi_path, msk_path, self.output_path)
        return
