import sys
import math
import numpy as np
import cv2

import torch
import torch.utils.data as data

# import openslide as ops
# import glymur

import matplotlib.pyplot as plt

import psutil

####
class SerializeFileList(data.IterableDataset):
    """
    Read a single file as multiple patches of same shape, perform the padding beforehand
    """
    def __init__(self, img_list, patch_info_list, patch_size):
        super(SerializeFileList).__init__()
        self.patch_size = patch_size

        self.img_list = img_list
        self.patch_info_list = patch_info_list

        self.worker_start_img_idx = 0
        # * for internal worker state
        self.stop_img_idx = 0
        self.curr_img_idx = 0
        self.curr_patch_idx = 0
        self.stop_patch_idx = 0
        return

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            print('hereX')
            self.stop_img_idx = len(self.img_list)
            return self
        else: # in a worker process so split workload, return a reduced copy of self
            # print('hereY', len(self.img_list), len(self.patch_info_list))
            per_worker = len(self.patch_info_list) / float(worker_info.num_workers)
            per_worker = int(math.ceil(per_worker))

            global_curr_patch_idx = worker_info.id * per_worker
            global_stop_patch_idx = global_curr_patch_idx + per_worker
            self.patch_info_list = self.patch_info_list[global_curr_patch_idx:global_stop_patch_idx]
            self.curr_patch_idx = 0
            self.stop_patch_idx = len(self.patch_info_list)
            # * check img indexer, implicit protocol in infer.py
            global_curr_img_idx = self.patch_info_list[0][-1]
            global_stop_img_idx = self.patch_info_list[-1][-1] + 1
            self.worker_start_img_idx = global_curr_img_idx
            self.img_list = self.img_list[global_curr_img_idx:global_stop_img_idx]
            self.curr_img_idx = 0
            self.stop_img_idx = len(self.img_list)
            return self # does it mutate source copy?

    def __next__(self):

        if self.curr_patch_idx >= self.stop_patch_idx:
            raise StopIteration # when there is nothing more to yield 
        patch_info = self.patch_info_list[self.curr_patch_idx]
        img_ptr = self.img_list[patch_info[-1] - self.worker_start_img_idx]
        patch_data = img_ptr[patch_info[0] : patch_info[0] + self.patch_size,
                             patch_info[1] : patch_info[1] + self.patch_size]
        self.curr_patch_idx += 1
        return patch_data, patch_info

####
class SerializeArray(data.Dataset):

    def __init__(self, image, patch_info_list, patch_size):
        super(SerializeArray).__init__()
        self.patch_size = patch_size

        self.image = image
        self.patch_info_list = patch_info_list
        return

    def __len__(self):
        return len(self.patch_info_list)

    def __getitem__(self, idx):
        patch_info = self.patch_info_list[idx]
        patch_data = self.image[patch_info[0] : patch_info[0] + self.patch_size[0],
                                patch_info[1] : patch_info[1] + self.patch_size[1]]    
        # print(patch_data.shape, patch_info[:2])
        return patch_data, patch_info

####
def visualize(ds, batch_size, nr_steps=100):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds) - 1:
            data_idx = 0
        nr_feed = len(ds[0])

        pos_counter = 0
        for j in range(0, batch_size):
            sample = ds[data_idx + j]
            for k in range(1, nr_feed + 1):
                plt.subplot(nr_feed, batch_size, k + j * nr_feed)
                plt.imshow(np.squeeze(sample[k - 1]))
        plt.show()
        data_idx += batch_size
