import multiprocessing as mp
from concurrent.futures import (FIRST_EXCEPTION, ProcessPoolExecutor,
                                as_completed, wait)

mp.set_start_method("spawn", True)  # ! must be at top for VScode debugging

import collections
import glob
import json
import logging
import math
import os
import pathlib
import copy
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
import torch.multiprocessing as torch_mp
import torch.utils.data as data
import tqdm

from misc.utils import (cropping_center, get_bounding_box, log_debug, log_info,
                        rm_n_mkdir)
from misc.wsi_handler import get_file_handler

from . import base


####
class SerializeArray(data.Dataset):
    """
    `mp_shared_space` must be from torch.multiprocessing, for example

    mp_manager = torch_mp.Manager()
    mp_shared_space = mp_manager.Namespace()
    mp_shared_space.image = torch.from_numpy(image)
    """
    def __init__(self, mp_shared_space, preproc=None):
        super().__init__()
        self.mp_shared_space = mp_shared_space
        self.preproc = preproc
        return

    def __len__(self):
        return len(self.mp_shared_space.patch_info_list)

    def __getitem__(self, idx):
        patch_info = self.mp_shared_space.patch_info_list[idx]
        tl, br = patch_info[0] # retrieve input placement, [1] is output
        patch_data = self.mp_shared_space.tile_img[tl[0] : br[0], tl[1] : br[1]]
        if self.preproc is not None:
            patch_dat = patch_data.copy()
            patch_data = self.preproc(patch_data)
        return patch_data, patch_info

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
    # print(info_list.shape)
    return info_list

####
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

    # * Full Tile Grid
    info_list = get_info_stack(output_tl, output_br).astype(np.int64)

    # flag surrounding ambiguous (left margin, right margin)
    # |----|------------|----|
    # |\\\\\\\\\\\\\\\\\\\\\\|  
    # |\\\\              \\\\|
    # |\\\\              \\\\|
    # |\\\\              \\\\|
    # |\\\\\\\\\\\\\\\\\\\\\\|  
    # |----|------------|----|
    removal_flag = np.full((info_list.shape[0], 4,), 1) # left, right, top, bot
    # exclude those contain left most boundary
    removal_flag[(info_list[:,1,0,1] == np.min(output_tl[:,1])),0] = 0
    # exclude those contain right most boundary
    removal_flag[(info_list[:,1,1,1] == np.max(output_br[:,1])),1] = 0
    # exclude those contain top most boundary
    removal_flag[(info_list[:,1,0,0] == np.min(output_tl[:,0])),2] = 0
    # exclude those contain bot most boundary
    removal_flag[(info_list[:,1,1,0] == np.max(output_br[:,0])),3] = 0
    mode_list = np.full(info_list.shape[0], 0)
    all_info = [[info_list, removal_flag, mode_list]]

    br_most = np.max(output_br, axis=0)
    tl_most = np.min(output_tl, axis=0)
    # * Tile Boundary Redo with Margin
    # get the fix grid tile info
    y_fix_output_tl = output_tl - np.array([margin_size[0], 0])[None,:]
    y_fix_output_br = np.stack([output_tl[:,0], output_br[:,1]], axis=-1)
    y_fix_output_br = y_fix_output_br + np.array([margin_size[0], 0])[None,:]
    # bound reassignment
    # ? do we need to do bound check for tl ? (extreme case of 1 tile of size < margin size ?)
    y_fix_output_br[y_fix_output_br[:,0] > br_most[0], 0] = br_most[0]  
    y_fix_output_br[y_fix_output_br[:,1] > br_most[1], 1] = br_most[1]  
    # sel position not on the image boundary
    sel = (output_tl[:,0] == np.min(output_tl[:,0]))
    y_info_list = get_info_stack(y_fix_output_tl[~sel], y_fix_output_br[~sel]).astype(np.int64)

    # flag horizontal ambiguous region for y (left margin, right margin)
    # |----|------------|----|
    # |\\\\|            |\\\\|  
    # |----|------------|----|
    # <----> ambiguous (margin size)
    removal_flag = np.zeros((y_info_list.shape[0], 4,)) # left, right, top, bot
    removal_flag[:,[0,1]] = 1
    # exclude the left most boundary
    removal_flag[(y_info_list[:,1,0,1] == np.min(output_tl[:,1])),0] = 0
    # exclude the right most boundary   
    removal_flag[(y_info_list[:,1,1,1] == np.max(output_br[:,1])),1] = 0
    mode_list = np.full(y_info_list.shape[0], 1)
    all_info.append([y_info_list, removal_flag, mode_list])

    x_fix_output_br = output_br + np.array([0, margin_size[1]])[None,:]
    x_fix_output_tl = np.stack([output_tl[:,0], output_br[:,1]], axis=-1)
    x_fix_output_tl = x_fix_output_tl - np.array([0, margin_size[1]])[None,:]
    # bound reassignment
    x_fix_output_br[x_fix_output_br[:,0] > br_most[0], 0] = br_most[0]  
    x_fix_output_br[x_fix_output_br[:,1] > br_most[1], 1] = br_most[1]  
    # sel position not on the image boundary
    sel = (output_br[:,1] == np.max(output_br[:,1]))
    x_info_list = get_info_stack(x_fix_output_tl[~sel], x_fix_output_br[~sel]).astype(np.int64)
    # flag vertical ambiguous region for x (top margin, bottom margin)
    # |----| ^
    # |\\\\| | ambiguous
    # |----| V
    # |    |
    # |    |
    # |----|
    # |\\\\|
    # |----|
    removal_flag = np.zeros((x_info_list.shape[0], 4,)) # left, right, top, bot
    removal_flag[:,[2,3]] = 1
    # exclude the left most boundary
    removal_flag[(x_info_list[:,1,0,0] == np.min(output_tl[:,0])),2] = 0
    # exclude the right most boundary   
    removal_flag[(x_info_list[:,1,1,0] == np.max(output_br[:,0])),3] = 0
    mode_list = np.full(x_info_list.shape[0], 2)
    all_info.append([x_info_list, removal_flag, mode_list])

    sel = np.any(output_br == br_most, axis=-1)
    xsect = output_br[~sel]
    xsect_tl = xsect - margin_size * 2
    xsect_br = xsect + margin_size * 2
    # do the bound check to ensure range stay within
    xsect_br[xsect_br[:,0] > br_most[0], 0] = br_most[0]  
    xsect_br[xsect_br[:,1] > br_most[1], 1] = br_most[1]  
    xsect_tl[xsect_tl[:,0] < tl_most[0], 0] = tl_most[0]  
    xsect_tl[xsect_tl[:,1] < tl_most[1], 1] = tl_most[1]  
    xsect_info_list = get_info_stack(xsect_tl, xsect_br).astype(np.int64)
    mode_list = np.full(xsect_info_list.shape[0], 3)
    removal_flag = np.full((xsect_info_list.shape[0], 4,), 0) # left, right, top, bot
    all_info.append([xsect_info_list, removal_flag, mode_list]) 

    return all_info
####
# ! seem to be 1 pix off at cross section or sthg
def get_inst_in_margin(arr, margin_size, tile_pp_info):
    """
    include the margin line itself
    """
    assert margin_size > 0
    tile_pp_info = np.array(tile_pp_info)

    inst_in_margin = []
    # extract those lie within margin region
    if tile_pp_info[0] == 1: # left edge
        inst_in_margin.append(arr[:,:(margin_size+1)])
    if tile_pp_info[1] == 1: # right edge
        inst_in_margin.append(arr[:,-(margin_size+1):])
    if tile_pp_info[2] == 1: # top edge
        inst_in_margin.append(arr[:(margin_size+1),:])
    if tile_pp_info[3] == 1: # bottom edge
        inst_in_margin.append(arr[-(margin_size+1):,:])
    inst_in_margin = [v.flatten() for v in inst_in_margin]
    if len(inst_in_margin) > 0:
        inst_in_margin = np.concatenate(inst_in_margin, axis=0)
        inst_in_margin = np.unique(inst_in_margin)
    else:
        inst_in_margin = np.array([]) # empty array
    return inst_in_margin

####
def get_inst_on_margin(arr, margin_size, tile_pp_info):
    """
    """
    assert margin_size > 0
    # extract those lie on the margin line
    tile_pp_info = np.array(tile_pp_info)

    def line_intersection(line1, line2):
        ydiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        xdiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0: 
            return False # not intersect

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        # ! positive region only (due to indexing line)
        return int(abs(y)), int(abs(x))

    last_h, last_w = arr.shape[0]-1, arr.shape[1]-1
    line_list = [
        [[0, 0]     , [last_h, 0]     ], # left line
        [[0, last_w], [last_h, last_w]], # right line
        [[0, 0]     , [0, last_w]     ], # top line
        [[last_h, 0], [last_h, last_w]], # bottom line
    ]

    if tile_pp_info[0] == 1: 
        line_list[0] = [[0     , margin_size], 
                       [last_h, margin_size]] 
    if tile_pp_info[1] == 1: 
        line_list[1] = [[0     , last_w-margin_size], 
                        [last_h, last_w-margin_size]]
    if tile_pp_info[2] == 1:
        line_list[2] = [[margin_size, 0], 
                        [margin_size, last_w]] 
    if tile_pp_info[3] == 1:
        line_list[3] = [[last_h-margin_size, 0], 
                        [last_h-margin_size, last_w]] 
    # x1 x2
    # x3 x4
    # all pts need to be valid idx !
    pts_list = [
        line_intersection(line_list[2], line_list[0]), # x1 
        line_intersection(line_list[2], line_list[1]), # x2
        line_intersection(line_list[3], line_list[0]), # x3
        line_intersection(line_list[3], line_list[1]), # x4
    ]

    pt_index = lambda p1, p2: arr[p1[0]:p2[0]+1,
                                  p1[1]:p2[1]+1]                        
    line_pix_list = []
    if tile_pp_info[0] == 1:
        line_pix_list.append(pt_index(pts_list[0], pts_list[2])),
    if tile_pp_info[1] == 1:
        line_pix_list.append(pt_index(pts_list[1], pts_list[3])),
    if tile_pp_info[2] == 1:
        line_pix_list.append(pt_index(pts_list[0], pts_list[1])),
    if tile_pp_info[3] == 1:
        line_pix_list.append(pt_index(pts_list[2], pts_list[3])),

    inst_on_margin = [v.flatten() for v in line_pix_list]

    if len(inst_on_margin) > 0:
        inst_on_margin = np.concatenate(inst_on_margin, axis=0)
        inst_on_margin = np.unique(inst_on_margin)
    else:
        inst_on_margin = np.array([]) # empty array

    return inst_on_margin
####
def get_inst_on_edge(arr, tile_pp_info):
    inst_on_edge = []
    if tile_pp_info[0] == 1:
        inst_on_edge.append(arr[:,0])
    if tile_pp_info[1] == 1:
        inst_on_edge.append(arr[:,-1])
    if tile_pp_info[2] == 1:
        inst_on_edge.append(arr[0,:])
    if tile_pp_info[3] == 1:
        inst_on_edge.append(arr[-1,:])

    inst_on_edge = [v.flatten() for v in inst_on_edge]

    if len(inst_on_edge) > 0:
        inst_on_edge = np.concatenate(inst_on_edge, axis=0)
        inst_on_edge = np.unique(inst_on_edge)
    else:
        inst_on_edge = np.array([]) # empty array
    return inst_on_edge
####

####
def run_model_forward(
        forward_output_queue,
        tile_info_list,
        wsi_path, wsi_ext, wsi_proc_mag, wsi_cache_path,
        run_step, model, loader_kwargs):

    wsi_handler = get_file_handler(wsi_path, backend=wsi_ext)
    wsi_handler.prepare_reading(read_mag=wsi_proc_mag)

    # using shared memory namespace so all the loader workers use same 
    # underlying image data, also allow persistent worker and fast data switching 
    mp_manager = torch_mp.Manager()
    mp_shared_space = mp_manager.Namespace()

    ds = SerializeArray(mp_shared_space)
    loader = data.DataLoader(ds, **loader_kwargs,
                            drop_last=False,
                            persistent_workers=True,
                        )

    all_time = 0
    for tile_idx, tile_info in enumerate(tile_info_list):

        tile_info, patch_info_list = tile_info

        tile_input_info = tile_info[0]
        tile_input_tl, tile_input_br = tile_input_info
        # shift from wsi system to tile input system
        # ! (tile output is within tile input system)
        # ! this will shift both patch input and output placement to tile input system
        # ! hence, output placement need to be shifted (corrected) later for post proc
        patch_info_list -= np.reshape(tile_input_tl, [1, 1, 1, 2])

        tile_img = wsi_handler.read_region(tile_input_tl[::-1], 
                                (tile_input_br - tile_input_tl)[::-1])

        # change the data in namespace to sync across persistent loader worker
        # also no need to do locking as these are assumed to be read only from worker
        start = time.perf_counter()
        mp_shared_space.tile_img = torch.from_numpy(tile_img).share_memory_()
        mp_shared_space.patch_info_list = torch.from_numpy(patch_info_list).share_memory_()
        end = time.perf_counter()
        all_time += (end - start)

        accumulated_patch_output = []
        for batch_idx, batch_data in enumerate(loader):
            sample_data_list, sample_info_list = batch_data
            sample_output_list = run_step(sample_data_list, model)
            sample_info_list = sample_info_list.numpy()
            accumulated_patch_output.append([sample_info_list, sample_output_list])
        forward_output_queue.put([tile_idx, accumulated_patch_output])
    log_info('Load Time: {0}'.format(all_time))
    return 

####
def postproc_tile(tile_io_info, tile_pp_info, tile_mode,
                margin_size, patch_info_list, func_opt,
                prev_wsi_inst_dict=None):
    # output pos of the tile within the source wsi
    tile_input_tl, tile_output_br = tile_io_info[0]
    tile_output_tl, tile_output_br = tile_io_info[1] # Y, X
    offset = tile_output_tl - tile_input_tl

    # ! shape may be uneven hence just detach all into a big list
    patch_pos_list  = [] 
    patch_feat_list = []
    split_inst = lambda x : np.split(x, x.shape[0], axis=0)
    for batch_pos, batch_feat in patch_info_list:
        patch_pos_list.extend(split_inst(batch_pos))
        patch_feat_list.extend(split_inst(batch_feat))

    # * assemble patch to tile
    nr_ch = patch_feat_list[-1].shape[-1]
    tile_shape = (tile_output_br - tile_output_tl).tolist()
    pred_map = np.zeros(tile_shape + [nr_ch], dtype=np.float32)
    for idx in range(len(patch_pos_list)):
        # zero idx to remove singleton, squeeze may kill h/w/c
        patch_pos = patch_pos_list[idx][0].copy()

        # # ! assume patch pos alrd aligned to be within tile input system
        patch_pos = patch_pos - offset # shift from wsi to tile output system

        pos_tl, pos_br = patch_pos[1] # retrieve ouput placement
        pred_map[
            pos_tl[0] : pos_br[0],
            pos_tl[1] : pos_br[1]
        ] = patch_feat_list[idx][0]
    del patch_pos_list, patch_feat_list

    # * retrieve actual output
    postproc_func, postproc_kwargs = func_opt
    pred_inst, inst_info_dict = postproc_func(pred_map, **postproc_kwargs)
    del pred_map

    # * perform removal for ambiguous region

    # Consider each symbol as 1 pixel    
    # This is margin inner area //// 
    # ----------------------------- ^  ^
    # |///////////////////////////| |  | margin area 
    # |///////////////////////////| |  | (yes including the inner and outer edge)
    # |///|-------------------|///| |  V
    # |///|   ^margin_size    |///| |
    # |///|                   |///| |
    # |///| <-- margin line   |///| | Image area
    # |///|      |            |///| |
    # |///|      v            |///| |
    # |///|-------------------|///| |
    # |///////////////////////////| |
    # |///////////////////////////| |
    # ----------------------------| V

    # ! extreme slow down may happen because the aggregated results are huge
    def draw_prev_pred_inst():
        tile_output = tile_io_info[1]
        tile_canvas = np.zeros(tile_output[1] - tile_output[0], dtype=np.int32)
        if len(prev_wsi_inst_dict) == 0: return tile_canvas

        wsi_inst_uid_list = np.array(list(prev_wsi_inst_dict.keys()))
        wsi_inst_com_list = np.array([v['centroid'] for v in prev_wsi_inst_dict.values()])
        wsi_inst_com_list = wsi_inst_com_list[:,::-1] # XY to YX       
        sel =  (wsi_inst_com_list[:,0] > tile_output[0,0])
        sel &= (wsi_inst_com_list[:,0] < tile_output[1,0])
        sel &= (wsi_inst_com_list[:,1] > tile_output[0,1])
        sel &= (wsi_inst_com_list[:,1] < tile_output[1,1])
        sel_idx = np.nonzero(sel.flatten())[0]

        for inst_idx in sel_idx:
            inst_uid = wsi_inst_uid_list[inst_idx]
            # shift from wsi system to tile output system
            inst_info = prev_wsi_inst_dict[inst_uid]
            inst_cnt = np.array(inst_info['contour'])
            inst_cnt = inst_cnt - tile_output[0,[1,0]][None]
            tile_canvas = cv2.drawContours(tile_canvas, [inst_cnt], -1, int(inst_uid), -1)
        return tile_canvas

    output_remove_inst_set = None
    # tile_mode = -1 # ! no fix, debud mode
    if tile_mode == 0: 
        # for `full grid tile`
        # -- extend from the boundary by the margin size, remove 
        #    nuclei lie within the margin area but exclude those
        #    lie on the margin line
        # also contain those lying on the edges
        inst_in_margin = get_inst_in_margin(pred_inst, margin_size, tile_pp_info)
        # those lying on the margin line, check 2pix toward margin area for sanity
        inst_on_margin1 = get_inst_on_margin(pred_inst, margin_size-1, tile_pp_info) 
        inst_on_margin2 = get_inst_on_margin(pred_inst, margin_size  , tile_pp_info) 
        inst_on_margin = np.union1d(inst_on_margin1, inst_on_margin2)
        inst_within_margin = np.setdiff1d(inst_in_margin, inst_on_margin, assume_unique=True)        
        remove_inst_set = inst_within_margin.tolist()
    elif tile_mode == 1 or tile_mode == 2:
        # for `horizontal/vertical strip tiles` for fixing artifacts
        # -- extend from the marked edges (top/bot or left/right) by the margin size, 
        #    remove all nuclei lie within the margin area (including on the margin line)
        # -- nuclei on all edges are removed (as these are alrd within `full grid tile`)
        inst_in_margin = get_inst_in_margin(pred_inst, margin_size, tile_pp_info) # also contain those lying on the edges
        if np.sum(tile_pp_info) == 1:
            holder_flag = tile_pp_info.copy()
            if tile_mode == 1: 
                holder_flag[[2, 3]] = 1
            else: 
                holder_flag[[0, 1]] = 1
        else:
            holder_flag = [1, 1, 1, 1]
        inst_on_edge = get_inst_on_edge(pred_inst, holder_flag)
        remove_inst_set = np.union1d(inst_in_margin, inst_on_edge)   
        remove_inst_set = remove_inst_set.tolist()
    elif tile_mode == 3:
        # inst within the tile after excluding margin area out
        # only for a tile at cross-section, which is designed such that 
        # their shape >= 3* margin size     

        # ! for removal of current pred, just like tile_mode = 0 but with all side   
        inst_in_margin = get_inst_in_margin(pred_inst, margin_size, [1, 1, 1, 1])
        # those lying on the margin line, check 2pix toward margin area for sanity
        inst_on_margin1 = get_inst_on_margin(pred_inst, margin_size-1, [1, 1, 1, 1]) 
        inst_on_margin2 = get_inst_on_margin(pred_inst, margin_size  , [1, 1, 1, 1]) 
        inst_on_margin = np.union1d(inst_on_margin1, inst_on_margin2)
        inst_within_margin = np.setdiff1d(inst_in_margin, inst_on_margin, assume_unique=True)        
        remove_inst_set = inst_within_margin.tolist()
        # ! but we also need to remove prev inst exist in the global space of entire wsi
        # ! on the margin
        prev_pred_inst = draw_prev_pred_inst()
        inst_on_margin1 = get_inst_on_margin(prev_pred_inst, margin_size-1, [1, 1, 1, 1]) 
        inst_on_margin2 = get_inst_on_margin(prev_pred_inst, margin_size  , [1, 1, 1, 1]) 
        inst_on_margin = np.union1d(inst_on_margin1, inst_on_margin2)
        output_remove_inst_set = inst_on_margin.tolist()    
    else:
        remove_inst_set = []

    remove_inst_set = set(remove_inst_set)
    # * move pos back to wsi position
    renew_id = 0
    new_inst_info_dict = {}
    for k, v in inst_info_dict.items():
        if k not in remove_inst_set:            
            v['bbox'] += tile_output_tl[::-1]
            v['centroid'] += tile_output_tl[::-1]
            v['contour'] += tile_output_tl[::-1]
            new_inst_info_dict[renew_id] = v
            renew_id += 1

    return new_inst_info_dict, output_remove_inst_set
####
class InferManager(base.InferManager):

    def __get_valid_patch_idx(self, patch_info_list):
        """Select valid patches from the list of input patch information.

        Args:
            patch_info_list: patch input coordinate information
            has_output_info: whether output information is given
        
        """
        def check_valid(info, wsi_mask):
            output_bbox = np.rint(info[1]).astype(np.int64)
            output_roi = wsi_mask[
                output_bbox[0][0] : output_bbox[1][0],
                output_bbox[0][1] : output_bbox[1][1],
            ]
            return (torch.sum(output_roi) > 0).item()

        down_sample_ratio = self.wsi_mask.shape[0] / self.wsi_proc_shape[0]
        torch_mask = torch.from_numpy(self.wsi_mask).share_memory_()
        valid_indices = [check_valid(info * down_sample_ratio, torch_mask) 
                         for info in patch_info_list]
        # somehow multiproc is slower than single thread
        valid_indices = np.array(valid_indices)
        return valid_indices

    def __get_valid_patch_idx_in_tile(self, tile_info, patch_info_list):
        # checking basing on the output alignment
        tile_tl, tile_br = tile_info[1]
        patch_tl_list = patch_info_list[:,1,0] 
        patch_br_list = patch_info_list[:,1,1]
        sel =  (patch_tl_list[:,0] >= tile_tl[0]) & (patch_tl_list[:,1] >= tile_tl[1])
        sel &= (patch_br_list[:,0] <= tile_br[0]) & (patch_br_list[:,1] <= tile_br[1])
        return sel     

    def _parse_args(self, run_args):
        """Parse command line arguments and set as instance variables."""
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        # to tuple
        make_shape_array = lambda x : np.array([x, x]).astype(np.int64)
        self.tile_shape = make_shape_array(self.tile_shape)
        self.ambiguous_size = make_shape_array(self.ambiguous_size)
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

            # ! using entire wsi, debugging purpose !
            # wsi_mask = np.array(simple_get_mask() > 0, dtype=np.uint8)
            wsi_mask = simple_get_mask()
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
        start = time.perf_counter()
        self.wsi_handler = get_file_handler(wsi_path, backend=wsi_ext)
        self.wsi_proc_shape = self.wsi_handler.get_dimensions(self.proc_mag)
        # ! cache here is for legacy and to deal with esoteric internal wsi format
        self.wsi_handler.prepare_reading(read_mag=self.proc_mag)
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

        # * retrieve patch and tile placement
        patch_info_list = _get_patch_info(
            self.wsi_proc_shape, self.patch_input_shape, self.patch_output_shape,
        )
        patch_diff_shape = self.patch_input_shape - self.patch_output_shape
        # derive tile output placement as consecutive tiling with step size of 0
        # and tile output will have shape being of multiple of patch_output_shape (round down)
        tile_output_shape = np.floor(self.tile_shape / self.patch_output_shape) * self.patch_output_shape
        tile_input_shape = tile_output_shape + patch_diff_shape
        
        # [full_grid, vert/horiz, xsect]
        all_tile_info = _get_tile_info(
            self.wsi_proc_shape, tile_input_shape, tile_output_shape, 
            self.ambiguous_size, self.patch_output_shape
        )
        end = time.perf_counter()
        log_info("Preparing Input Output Placement: {0}".format(end - start))

        # * Async Inference
        # * launch a seperate process to do forward and store the result in a queue.
        # * main thread will poll for forward result and launch separate process for
        # * doing the postproc for every newly predicted tiles
        #
        #         / forward \ (loop)
        # main ------------- main----------------main
        #                       \ postproc (loop)/

        sel_index = self.__get_valid_patch_idx(patch_info_list)
        patch_info_list = patch_info_list[sel_index]
        
        pbar_creator = lambda x, y, pos=0, leave=False: tqdm.tqdm(
            desc=x, leave=leave, total=y, ncols=80, ascii=True, position=pos
        )

        def run_once(tile_io_info_list, tile_pp_info_list, tile_mode_list,
                     prev_wsi_inst_dict=None):

            sel_index = self.__get_valid_patch_idx(tile_io_info_list)
            tile_io_info_list = tile_io_info_list[sel_index]
            tile_pp_info_list = tile_pp_info_list[sel_index]
            tile_mode_list    = tile_mode_list[sel_index]
            
            mp_manager = torch_mp.Manager()
            # contain at most 5 tile ouput before polling forward func
            mp_forward_output_queue = mp_manager.Queue(maxsize=16)

            forward_info_list = collections.deque()
            nr_tile = tile_io_info_list.shape[0]
            for tile_idx in range(nr_tile):
                tile_info = tile_io_info_list[tile_idx]
                sel_index = self.__get_valid_patch_idx_in_tile(tile_info, patch_info_list)
                patch_in_tile_info_list = patch_info_list[sel_index]
                forward_info_list.append([tile_info, patch_in_tile_info_list])

            loader_kwargs = dict(
                num_workers=self.nr_inference_workers,
                batch_size=self.batch_size,
            )

            wsi_cache_path = "%s/src_wsi.npy" % self.cache_path
            forward_process = mp.Process(target=run_model_forward, 
                                        args=(mp_forward_output_queue, forward_info_list,
                                                wsi_path, wsi_ext, self.proc_mag, wsi_cache_path,
                                                self.run_step, self.model, loader_kwargs))
            forward_process.start()

            post_proc_kwargs = {
                "nr_types": self.method["model_args"]["nr_types"],
                "return_centroids": True,
            }

            proc_pool = None
            if self.nr_post_proc_workers > 0:
                proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

            offset_id = 0 # offset id to increment overall saving dict
            wsi_inst_info = {} 
            if prev_wsi_inst_dict is not None:
                wsi_inst_info = copy.deepcopy(prev_wsi_inst_dict)
                if len(wsi_inst_info) > 0:
                    offset_id = max(wsi_inst_info.keys()) + 1

            foward_pbar = pbar_creator('Forward', nr_tile, pos=0)
            postpr_pbar = pbar_creator('PostPro', nr_tile, pos=1)

            future_list = collections.deque()
            # will this lead to infinite loop ?
            while True:
                if forward_process.exitcode is not None \
                    and mp_forward_output_queue.empty():
                    break
                
                if not mp_forward_output_queue.empty():
                    tile_idx, forward_output = mp_forward_output_queue.get()
                    foward_pbar.update()

                    tile_io_info = tile_io_info_list[tile_idx]
                    tile_pp_info = tile_pp_info_list[tile_idx]
                    tile_mode = tile_mode_list[tile_idx]
                    args = [tile_io_info, tile_pp_info, tile_mode, 
                            self.ambiguous_size[0], forward_output, 
                            (self.post_proc_func, post_proc_kwargs),
                            prev_wsi_inst_dict]

                    if proc_pool is not None:
                        future = proc_pool.submit(postproc_tile, *args)
                        future_list.append(future)
                    else:
                        new_inst_dict, remove_uid_list = postproc_tile(*args)
    
                        # * aggregate
                        # ! the return id should be contiguous to maximuize 
                        # ! counting range in int32
                        inst_wsi_id = offset_id # barrier in case no output in tile!
                        for inst_id, inst_info in new_inst_dict.items():
                            inst_wsi_id = offset_id + inst_id + 1
                            wsi_inst_info[inst_wsi_id] = inst_info
                        offset_id = inst_wsi_id + 1

                        if remove_uid_list is not None:
                            for inst_uid in remove_uid_list:
                                if inst_uid in wsi_inst_info:
                                    # faster than pop
                                    del wsi_inst_info[inst_uid]
                        postpr_pbar.update()

            if forward_process.exitcode > 0:
                raise ValueError(f'Forward process exited with code {forward_process.exitcode}')
            forward_process.join()

            while len(future_list) > 0:
                if not future_list[0].done(): 
                    future_list.rotate()
                    continue
                proc_future = future_list.popleft()
                if proc_future.exception() is not None:
                    print(proc_future.exception())

                # * aggregate
                # ! the return id should be contiguous to maximize 
                # ! counting range in int32
                new_inst_dict, remove_uid_list = proc_future.result()
                inst_wsi_id = offset_id # barrier in case no output in tile!
                for inst_id, inst_info in new_inst_dict.items():
                    inst_wsi_id = offset_id + inst_id + 1
                    wsi_inst_info[inst_wsi_id] = inst_info
                offset_id = inst_wsi_id + 1

                if remove_uid_list is not None:
                    for inst_uid in remove_uid_list:
                        if inst_uid in wsi_inst_info:
                            # faster than pop
                            del wsi_inst_info[inst_uid]
                postpr_pbar.update()
            foward_pbar.close()
            postpr_pbar.close()
            proc_pool.shutdown()
            return wsi_inst_info

        #
        ## ** process full grid and vert/horiz fixing at the same time
        start = time.perf_counter()
        info_list = list(zip(*all_tile_info[:3]))
        tile_io_info_list = np.concatenate(info_list[0], axis=0).astype(np.int32)
        tile_pp_info_list = np.concatenate(info_list[1], axis=0)
        tile_mode_list    = np.concatenate(info_list[2], axis=0)
        wsi_inst_info = run_once(tile_io_info_list, 
                                 tile_pp_info_list, 
                                 tile_mode_list)
        end = time.perf_counter()
        log_info("Proc Grid: {0}".format(end - start))
        
        # wsi_inst_info = {}
        ## ** re-infer and redo postproc for xsect alone
        start = time.perf_counter()
        tile_io_info_list = all_tile_info[-1][0]
        tile_pp_info_list = all_tile_info[-1][1]
        tile_mode_list    = all_tile_info[-1][2]
        wsi_inst_info = run_once(tile_io_info_list, 
                                 tile_pp_info_list, 
                                 tile_mode_list, 
                                 wsi_inst_info)          
        end = time.perf_counter()
        log_info("Proc XSect: {0}".format(end - start))

        if self.save_mask or self.save_thumb:
            json_path = "%s/json/%s.json" % (output_dir, wsi_name)
        else:
            json_path = "%s/%s.json" % (output_dir, wsi_name)
        self.__save_json(json_path, wsi_inst_info)
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
        for wsi_path in wsi_path_list:
            wsi_base_name = pathlib.Path(wsi_path).stem

            msk_path = "%s/%s.png" % (self.input_mask_dir, wsi_base_name)
            if self.save_thumb or self.save_mask:
                output_file = "%s/json/%s.json" % (self.output_dir, wsi_base_name)
            else:
                output_file = "%s/%s.json" % (self.output_dir, wsi_base_name)

            # ! cache in case of loading across network
            # start = time.perf_counter()
            # wsi_holder_path = self.cache_src_wsi_path
            # shutil.copyfile(wsi_path, wsi_holder_path)
            # end = time.perf_counter()
            # log_info('Move WSI: {0}'.format(end - start))

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
