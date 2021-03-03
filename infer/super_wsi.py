import multiprocessing as mp
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from multiprocessing import Lock, Pool

mp.set_start_method("spawn", True)  # ! must be at top for VScode debugging

import argparse
import glob
import json
import logging
import math
import os
import pathlib
import re
import shutil
import sys
import time
from functools import reduce
from importlib import import_module

import cv2
import numpy as np
import psutil
import scipy.io as sio
import torch
import torch.utils.data as data
import tqdm
from docopt import docopt

# from dataloader.infer_loader import SerializeArray, SerializeFileList
# from misc.utils import (
#     cropping_center,
#     get_bounding_box,
#     log_debug,
#     log_info,
#     rm_n_mkdir,
# )
# from misc.wsi_handler import get_file_handler
# from . import base


####
def _remove_inst(inst_map, remove_id_list):
    """Remove instances with id in remove_id_list.
    
    Args:
        inst_map: map of instances
        remove_id_list: list of ids to remove from inst_map
    """
    for inst_id in remove_id_list:
        inst_map[inst_map == inst_id] = 0
    return inst_map


####
def _get_patch_info(img_shape, input_size, output_size):
    """Get top left coordinate information of patches from original image.

    Args:
        img_shape: input image shape
        input_size: patch input shape
        output_size: patch output shape

    """
    def flat_mesh_grid_coord(y, x):
        y, x = np.meshgrid(y, x)
        return np.stack([y.flatten(), x.flatten()], axis=-1)

    in_out_diff = input_size - output_size
    nr_step = np.floor((img_shape - in_out_diff) / output_size) + 1
    last_output_coord = (in_out_diff // 2) + (nr_step) * output_size
    # generating subpatches index from orginal
    output_tl_y_list = np.arange(
        in_out_diff[0] // 2, last_output_coord[0], output_size[0], dtype=np.int32
    )
    output_tl_x_list = np.arange(
        in_out_diff[1] // 2, last_output_coord[1], output_size[1], dtype=np.int32
    )
    output_tl = flat_mesh_grid_coord(output_tl_y_list, output_tl_x_list)
    output_br = output_tl + output_size

    input_tl = output_tl - in_out_diff // 2
    input_br = input_tl + input_size
    # exclude any patch where the input exceed the range of image,
    # can comment this out if do padding in reading
    sel = np.any(input_br > img_shape, axis=-1)

    info_list = np.stack(
        [
            np.stack([ input_tl[~sel],  input_br[~sel]], axis=1),
            np.stack([output_tl[~sel], output_br[~sel]], axis=1),
        ], axis=1)
    print(info_list.shape)
    return info_list

# info_list = _get_patch_info(
#                 np.array([70, 100]), 
#                 np.array([40, 40]), 
#                 np.array([20, 20]))
# print(info_list[:,1,0])

# info_list = _get_patch_info(
#                 np.array([100, 100]), 
#                 np.array([80, 80]), 
#                 np.array([60, 60]))
# print(info_list[:,0,1])
# exit()

def _get_tile_info(img_shape, input_size, output_size, margin_size, unit_size):
    """Get top left coordinate information of patches from original image.

    Args:
        img_shape: input image shape
        input_size: patch input shape
        output_size: patch output shape

    """
    # ! ouput tile size must be multiple of unit
    assert np.sum(output_size % unit_size) == 0
    assert np.sum((margin_size*2) % unit_size) == 0

    def flat_mesh_grid_coord(y, x):
        y, x = np.meshgrid(y, x)
        return np.stack([y.flatten(), x.flatten()], axis=-1)

    in_out_diff = input_size - output_size
    nr_step = np.ceil((img_shape - in_out_diff) / output_size)
    last_output_coord = (in_out_diff // 2) + (nr_step) * output_size

    assert np.sum(output_size % unit_size) == 0
    nr_unit_step = np.floor((img_shape - in_out_diff) / unit_size)
    last_unit_output_coord = (in_out_diff // 2) + (nr_unit_step) * unit_size

    # generating subpatches index from orginal
    def get_top_left_1d(axis):
        o_tl_list = np.arange(
                        in_out_diff[axis] // 2, 
                        last_output_coord[axis], 
                        output_size[axis], dtype=np.int32
                    )
        o_br_list = o_tl_list + output_size[axis]
        o_br_list[-1] = last_unit_output_coord[axis]
        # in default behavior, last pos >= last multiple of unit
        # hence may cause duplication, do a check and remove if necessary
        if o_br_list[-1] == o_br_list[-2]:
            o_br_list = o_br_list[:-1]
            o_tl_list = o_tl_list[:-1]
        return o_tl_list, o_br_list

    output_tl_y_list, output_br_y_list = get_top_left_1d(axis=0)
    output_tl_x_list, output_br_x_list = get_top_left_1d(axis=1)

    output_tl = flat_mesh_grid_coord(output_tl_y_list, output_tl_x_list)
    output_br = flat_mesh_grid_coord(output_br_y_list, output_br_x_list)

    def get_info_stack(output_tl, output_br):
        input_tl = output_tl - (in_out_diff // 2)
        input_br = output_br + (in_out_diff // 2)

        info_list = np.stack(
            [
                np.stack([ input_tl,  input_br], axis=1),
                np.stack([output_tl, output_br], axis=1),
            ], axis=1)
        return info_list
    info_list = get_info_stack(output_tl, output_br)

    br_most = np.max(output_br, axis=0)
    # get the fix grid tile info
    y_fix_output_tl = output_tl - np.array([margin_size[0], 0])[None,:]
    y_fix_output_br = np.stack([output_tl[:,0], output_br[:,1]], axis=-1)
    y_fix_output_br = y_fix_output_br + np.array([margin_size[0], 0])[None,:]
    # bound reassignment
    y_fix_output_br[y_fix_output_br[:,0] > br_most[0], 0] = br_most[0]  
    y_fix_output_br[y_fix_output_br[:,1] > br_most[1], 1] = br_most[1]  
    # sel position not on the image boundary
    sel = (output_tl[:,0] == np.min(output_tl[:,0]))
    y_info_list = get_info_stack(y_fix_output_tl[~sel], y_fix_output_br[~sel])
    # print(y_info_list[...,::-1][:,1],'\n')

    # flag horizontal ambiguous region for y (left margin, right margin)
    # |----|------------|----|
    # |\\\\|            |\\\\|  
    # |----|------------|----|
    # ambiguous         ambiguous (margin size)
    removal_flag = np.zeros((y_info_list.shape[0], 4,)) # left, right, top, bot
    removal_flag[:,[0,1]] = 1
    # exclude the left most boundary
    removal_flag[(y_info_list[:,1,0,1] == np.min(output_tl[:,1])),0] = 0
    # exclude the right most boundary   
    removal_flag[(y_info_list[:,1,1,1] == np.max(output_br[:,1])),1] = 0
    print(removal_flag)
    print(y_info_list[...,::-1][:,1])

    x_fix_output_br = output_br + np.array([0, margin_size[1]])[None,:]
    x_fix_output_tl = np.stack([output_tl[:,0], output_br[:,1]], axis=-1)
    x_fix_output_tl = x_fix_output_tl - np.array([0, margin_size[1]])[None,:]
    # bound reassignment
    x_fix_output_br[x_fix_output_br[:,0] > br_most[0], 0] = br_most[0]  
    x_fix_output_br[x_fix_output_br[:,1] > br_most[1], 1] = br_most[1]  
    # sel position not on the image boundary
    sel = (output_br[:,1] == np.max(output_br[:,1]))
    x_info_list = get_info_stack(x_fix_output_tl[~sel], x_fix_output_br[~sel])
    # print(x_info_list[...,::-1][:,1],'\n')
    # flag vertical ambiguous region for x (top margin, bottom margin)
    # |----|
    # |\\\\| ambiguous
    # |----|
    # |    |
    # |    |
    # |----|
    # |\\\\| ambiguous
    # |----|
    removal_flag = np.zeros((x_info_list.shape[0], 4,)) # left, right, top, bot
    removal_flag[:,[2,3]] = 1
    # exclude the left most boundary
    removal_flag[(x_info_list[:,1,0,0] == np.min(output_tl[:,0])),2] = 0
    # exclude the right most boundary   
    removal_flag[(x_info_list[:,1,1,0] == np.max(output_br[:,0])),3] = 0
    print(removal_flag)
    print(x_info_list[...,::-1][:,1])

    return info_list

info_list = _get_tile_info(
                np.array([170, 100]), # [130, 100], [140, 100]
                np.array([80, 80]), 
                np.array([60, 60]), 
                np.array([20, 20]),
                np.array([20, 20]),)
# print(info_list[:,0])
exit()

####
def _get_tile_patch_info(
    img_shape, tile_shape, patch_input_shape, patch_output_shape
):
    """Get chunk patch info. Here, chunk refers to tiles used during inference.

    Args:
        img_shape: input image shape
        tile_input_shape: shape of tiles used for processing
        patch_input_shape: input patch shape
        patch_output_shape: output patch shape

    """
    def flat_mesh_grid_coord(y, x):
        y, x = np.meshgrid(y, x)
        return np.stack([y.flatten(), x.flatten()], axis=-1)

    patch_diff_shape = patch_input_shape - patch_output_shape
    patch_info_list = _get_patch_info(img_shape, 
                        patch_input_shape, patch_output_shape, 
                        drop_out_of_range=True)

    round_to_multiple = lambda x, y: np.floor(x / y) * y
    # derive tile output placement as consecutive tiling with step size of 0
    # and tile output will have shape of multiple of patch_output_shape (round down)
    tile_output_shape = int(tile_shape / patch_output_shape) * patch_output_shape
    tile_input_shape = tile_output_shape + patch_diff_shape
    tile_info_list = _get_patch_info(img_shape, 
                        patch_input_shape, patch_output_shape, 
                        drop_out_of_range=False)   
    tile_i_list = tile_info_list[:,0]
    tile_o_list = tile_info_list[:,1]

    return

####
class InferManager(base.InferManager):
    def __run_model(self, patch_top_left_list, pbar_desc):
        # TODO: the cost of creating dataloader may not be cheap ?
        dataset = SerializeArray(
            "%s/cache_chunk.npy" % self.cache_path,
            patch_top_left_list,
            self.patch_input_shape,
        )

        dataloader = data.DataLoader(
            dataset,
            num_workers=self.nr_inference_workers,
            batch_size=self.batch_size,
            drop_last=False,
        )

        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=int(len(dataloader)),
            ncols=80,
            ascii=True,
            position=0,
        )

        # run inference on input patches
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

    def __select_valid_patches(self, patch_info_list, has_output_info=True):
        """Select valid patches from the list of input patch information.

        Args:
            patch_info_list: patch input coordinate information
            has_output_info: whether output information is given
        
        """
        down_sample_ratio = self.wsi_mask.shape[0] / self.wsi_proc_shape[0]
        selected_indices = []
        for idx in range(patch_info_list.shape[0]):
            patch_info = patch_info_list[idx]
            patch_info = np.squeeze(patch_info)
            # get the box at corresponding mag of the mask
            if has_output_info:
                output_bbox = patch_info[1] * down_sample_ratio
            else:
                output_bbox = patch_info * down_sample_ratio
            output_bbox = np.rint(output_bbox).astype(np.int64)
            # coord of the output of the patch (i.e center regions)
            output_roi = self.wsi_mask[
                output_bbox[0][0] : output_bbox[1][0],
                output_bbox[0][1] : output_bbox[1][1],
            ]
            if np.sum(output_roi) > 0:
                selected_indices.append(idx)
        sub_patch_info_list = patch_info_list[selected_indices]
        return sub_patch_info_list

    def _parse_args(self, run_args):
        """Parse command line arguments and set as instance variables."""
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        # to tuple
        make_shape_array = lambda x : np.array([x, x]).astype(np.int64)
        self.tile_shape = make_shape_array(self.tile_shape)
        self.patch_input_shape = make_shape_array(self.patch_input_shape)
        self.patch_output_shape = make_shape_array(self.patch_output_shape)
        return

    def _get_wsi_mask(self, wsi_handler, mask_path):
        if mask_path is not None and os.path.isfile(mask_path):
            wsi_mask = cv2.imread(mask_path)
            wsi_mask = cv2.cvtColor(self.wsi_mask, cv2.COLOR_BGR2GRAY)
            wsi_mask[wsi_mask > 0] = 1
        else:
            log_info(
                "WARNING: No mask found, generating mask via thresholding at 1.25x!"
            )
            from skimage import morphology

            # simple method to extract tissue regions using intensity thresholding and morphological operations
            def simple_get_mask():
                scaled_wsi_mag = 1.25  # ! hard coded
                wsi_thumb_rgb = wsi_handler.get_full_img(read_mag=scaled_wsi_mag)
                gray = cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                mask = morphology.remove_small_objects(
                    mask == 0, min_size=16 * 16, connectivity=2
                )
                mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
                mask = morphology.binary_dilation(mask, morphology.disk(16))
                return mask

            wsi_mask = np.array(simple_get_mask() > 0, dtype=np.uint8)
        return wsi_mask

    def process_single_file(self, wsi_path, mask_path, output_dir):
        """Process a single whole-slide image and save the results.

        Args:
            wsi_path: path to input whole-slide image
            msk_path: path to input mask. If not supplied, mask will be automatically generated.
            output_dir: path where output will be saved

        """
        path_obj = pathlib.Path(wsi_path)
        wsi_ext = path_obj.suffix
        wsi_name = path_obj.stem

        # TODO: expose read mpp mode
        self.wsi_handler = get_file_handler(wsi_path, backend=wsi_ext)
        self.wsi_proc_shape = self.wsi_handler.get_dimensions(self.proc_mag)
        # ! cache here is for legacy and to deal with esoteric internal wsi format
        self.wsi_handler.prepare_reading(
            read_mag=self.proc_mag, cache_path="%s/src_wsi.npy" % self.cache_path
        )
        self.wsi_proc_shape = np.array(self.wsi_proc_shape[::-1])  # to Y, X

        self.wsi_mask = self._get_wsi_mask(self.wsi_handler, mask_path)
        if np.sum(self.wsi_mask) == 0:
            log_info("Skip due to empty mask!")
            return
        if self.save_mask:
            cv2.imwrite("%s/mask/%s.png" % (output_dir, wsi_name), 
                self.wsi_mask * 255)
        if self.save_thumb:
            wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=1.25)
            cv2.imwrite(
                "%s/thumb/%s.png" % (output_dir, wsi_name),
                cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2BGR),
            )

        # * raw prediction
        chunk_info_list, patch_info_list = _get_chunk_patch_info(
            self.wsi_proc_shape,
            chunk_input_shape,
            patch_input_shape,
            patch_output_shape,
        )

        return

    def process_wsi_list(self, run_args):
        """Process a list of whole-slide images.

        Args:
            run_args: arguments as defined in run_infer.py
        
        """
        self._parse_args(run_args)

        if not os.path.exists(self.cache_path):
            rm_n_mkdir(self.cache_path)

        if not os.path.exists(self.output_dir + "/json/"):
            rm_n_mkdir(self.output_dir + "/json/")
        if self.save_thumb:
            if not os.path.exists(self.output_dir + "/thumb/"):
                rm_n_mkdir(self.output_dir + "/thumb/")
        if self.save_mask:
            if not os.path.exists(self.output_dir + "/mask/"):
                rm_n_mkdir(self.output_dir + "/mask/")

        wsi_path_list = glob.glob(self.input_dir + "/*")
        wsi_path_list.sort()  # ensure ordering
        for wsi_path in wsi_path_list[:]:
            wsi_base_name = pathlib.Path(wsi_path).stem
            msk_path = "%s/%s.png" % (self.input_mask_dir, wsi_base_name)
            if self.save_thumb or self.save_mask:
                output_file = "%s/json/%s.json" % (self.output_dir, wsi_base_name)
            else:
                output_file = "%s/%s.json" % (self.output_dir, wsi_base_name)
            if os.path.exists(output_file):
                log_info("Skip: %s" % wsi_base_name)
                continue
            try:
                log_info("Process: %s" % wsi_base_name)
                self.process_single_file(wsi_path, msk_path, self.output_dir)
                log_info("Finish")
            except:
                logging.exception("Crash")
        rm_n_mkdir(self.cache_path)  # clean up all cache
        return
