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

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
import tqdm
import psutil
from dataloader.infer_loader import SerializeFileList, SerializeArray
from functools import reduce
import pickle

from misc.utils import rm_n_mkdir, cropping_center, get_bounding_box
from misc.viz_utils import visualize_instances_dict

from . import base

####
def _prepare_patching(img, window_size, mask_size, return_src_top_corner=False):
    win_size = window_size
    msk_size = step_size = mask_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = img.shape[0]
    im_w = img.shape[1]

    last_h, _ = get_last_steps(im_h, msk_size, step_size)
    last_w, _ = get_last_steps(im_w, msk_size, step_size)

    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - im_h
    padr = last_w + win_size - im_w

    img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

    # generating subpatches index from orginal
    coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
    coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
    row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    coord_y = coord_y.flatten() 
    coord_x = coord_x.flatten()
    row_idx = row_idx.flatten() 
    col_idx = col_idx.flatten()
    #
    patch_info = np.stack([coord_y, coord_x, 
                           row_idx, col_idx], axis=-1)
    if not return_src_top_corner:
        return img, patch_info
    else:
        return img, patch_info, [padt, padl]

#### 
def _post_process_patches(functor, patch_info, src_info, get_overlaid=False, type_colour=None, *args):
    # re-assemble the prediction, sort according to the patch location within the original image
    patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]]) 
    patch_info, patch_data = zip(*patch_info)

    src_shape = src_info['src_shape']
    src_image = src_info['src_image']

    patch_shape = np.squeeze(patch_data[0]).shape
    ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
    axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

    nr_row = max([x[2] for x in patch_info]) + 1
    nr_col = max([x[3] for x in patch_info]) + 1
    pred_map = np.concatenate(patch_data, axis=0)
    pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
    pred_map = np.transpose(pred_map, axes)
    pred_map = np.reshape(pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch))
    # crop back to original shape
    pred_map = np.squeeze(pred_map[:src_shape[0], :src_shape[1]])

    # * Implicit protocol
    # * a prediction map with instance of ID 1-N
    # * and a dict contain the instance info, access via its ID
    # * each instance may have type
    pred_inst, inst_info_dict = functor(pred_map) 

    overlaid_img = None
    if get_overlaid:
        overlaid_img = visualize_instances_dict(src_image, 
                                            inst_info_dict, 
                                            type_colour=type_colour)

    return [pred_inst, inst_info_dict , overlaid_img], args
    
class Inferer(base.Inferer):
    """
    Run inference on tiles
    """ 

    ####
    def process_file_list(self, 
                nr_inference_workers=4,
                nr_post_proc_workers=4,
                batch_size=None,
                input_dir=None,
                output_dir=None,
                patch_input_shape=None,
                patch_output_shape=None,
                save_intermediate_output=True,):
        """
        Process a single image tile < 5000x5000 in size.
        """
        # * depend on the number of samples and their size, this may be less efficient
        file_path_list = glob.glob('%s/*' % input_dir)
        file_path_list.sort()  # ensure same order
        file_path_list = file_path_list[:4]

        rm_n_mkdir(output_dir)

        def proc_callback(args):
            """
            output format is implicit assumption
            """
            results, args = args # TODO: args is not very intuitive
            pred_inst, inst_info_dict, overlaid = results

            base_name = args[0]
            if overlaid is not None:
                overlaid = cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR)
                cv2.imwrite('%s/%s.png' % (output_dir, base_name), overlaid)
            sio.savemat('%s/%s.mat' % (output_dir, base_name), {'inst_map': pred_inst})
            # TODO: human readable ? but JSON is not good for large files
            with open('%s/%s.pickle' % (output_dir, base_name), 'wb') as handle:
                pickle.dump(inst_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        def detach_items_of_uid(items_list, uid, nr_expected_items):            
            item_counter = 0
            detached_items_list = []
            remained_items_list = []
            while True: 
                pinfo, pdata = items_list.pop(0)
                pinfo = np.squeeze(pinfo)
                if pinfo[-1] == uid:
                    detached_items_list.append([pinfo, pdata])
                    item_counter += 1
                else:
                    remained_items_list.append([pinfo, pdata])
                if item_counter == nr_expected_items:
                    break
            # do this to ensure the ordering
            remained_items_list = remained_items_list + items_list
            return detached_items_list, remained_items_list

        proc_pool = None
        if nr_post_proc_workers != 0:            
            proc_pool = Pool(processes=nr_post_proc_workers)

        while len(file_path_list) > 0:
  
            hardware_stats = psutil.virtual_memory()
            available_ram = getattr(hardware_stats, 'available')
            available_ram = int(available_ram * 0.6)
            # available_ram >> 20 for MB, >> 30 for GB

            # TODO: this portion looks clunky but seems hard to detach into separate func

            # * caching N-files into memory such that their expected (total) memory usage 
            # * does not exceed the designated percentage of currently available memory
            # * the expected memory is a factor w.r.t original input file size and 
            # * must be manually provided
            file_idx = 0
            use_path_list = []
            cache_image_list = []
            cache_patch_info_list = []
            cache_image_info_list = []
            while len(file_path_list) > 0:
                file_path = file_path_list.pop(0)

                img = cv2.imread(file_path)
                src_shape = img.shape
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img, patch_info, top_corner = _prepare_patching(img, 
                                                    patch_input_shape, 
                                                    patch_output_shape, True)
                self_idx = np.full(patch_info.shape[0], file_idx, dtype=np.int32)
                patch_info = np.concatenate([patch_info, self_idx[:,None]], axis=-1)
                # ? may be expensive op
                patch_info = np.split(patch_info, patch_info.shape[0], axis=0) 
                patch_info = [np.squeeze(p) for p in patch_info]

                # * this factor=5 is only applicable for HoVerNet
                expected_usage = sys.getsizeof(img) * 5
                available_ram -= expected_usage
                if available_ram < 0: 
                    break
                
                file_idx += 1
                use_path_list.append(file_path)
                cache_image_list.append(img)
                cache_patch_info_list.extend(patch_info)
                # TODO: refactor to explicit protocol
                cache_image_info_list.append([src_shape, len(patch_info), top_corner])

            # * apply neural net on cached data
            dataset = SerializeFileList(cache_image_list, 
                                cache_patch_info_list, 
                                patch_input_shape)

            dataloader = data.DataLoader(dataset,
                                num_workers=nr_inference_workers,
                                batch_size=batch_size,
                                drop_last=False)

            pbar = tqdm.tqdm(desc='Process Patches', leave=True,
                        total=int(len(cache_patch_info_list) / batch_size) + 1, 
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
            
            # * parallely assemble the processed cache data for each file if possible
            for file_idx, file_path in enumerate(use_path_list):
                image_info = cache_image_info_list[file_idx]
                file_ouput_data, accumulated_patch_output = detach_items_of_uid(
                                                                accumulated_patch_output, 
                                                                file_idx, image_info[1])

                # * detach this into func and multiproc dispatch it
                src_pos = image_info[2] # src top left corner within padded image
                src_image = cache_image_list[file_idx]
                src_image = src_image[src_pos[0]:src_pos[0]+src_shape[0],
                                      src_pos[1]:src_pos[1]+src_shape[1]]
                file_info = {'src_shape' : image_info[0], 'src_image' : src_image}
                base_name = os.path.basename(file_path).split('.')[0]

                # TODO: retrieve the color ?
                func_args = [self.post_proc_func, file_ouput_data, file_info, True, None, base_name]

                # dispatch for parallel post-processing
                if proc_pool is not None:
                    proc_pool.apply_async(_post_process_patches, args=func_args, 
                                    callback=proc_callback)
                else:
                    proc_output = _post_process_patches(*func_args)
                    proc_callback(proc_output)

        if proc_pool is not None:
            proc_pool.close()
            proc_pool.join()
        return 
