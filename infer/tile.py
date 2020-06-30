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
from docopt import docopt
import tqdm
import psutil
from dataloader.infer_loader import SerializeFileList, SerializeArray
from functools import reduce

from misc.patch_extractor import prepare_patching
from misc.utils import rm_n_mkdir, cropping_center, get_bounding_box
from postproc import hover

from . import base


class InferTile(base.InferManager):
    """
    Run inference on tiles
    """ 
    ####
    def process_file_list(self, 
                        nr_worker=None, 
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

        def proc_callback(output, base_name):
            """
            output format is implicit assumption
            """
            pred_inst, pred_type, overlaid = output
            if overlaid is not None:
                cv2.imwrite('%s/%s.png' % (output_dir, base_name), overlaid)
            if self.type_classification:
                sio.savemat('%s/%s.mat' % (output_dir, base_name),
                            {'inst_map': pred_inst, 'type_map': pred_type})
            else:
                sio.savemat('%s/%s.mat' % (output_dir, base_name), 
                            {'inst_map': pred_inst})

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

        nr_procs = 2
        proc_pool = None if nr_procs == 0 else Pool(processes=nr_procs)

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
                img, patch_info, top_corner = prepare_patching(img, 
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
                                num_workers=nr_worker,
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
                func_args = [file_ouput_data, file_info]

                # dispatch for parallel post-processing
                if nr_procs > 0:
                    proc_pool.apply_async(post_process_patches, args=func_args, 
                                    callback=lambda x: proc_callback(x, base_name))
                else:
                    proc_output = post_process_patches(*func_args)
                    proc_callback(proc_output, base_name)

        if nr_procs > 0:
            proc_pool.close()
            proc_pool.join()
        return 
