
import math
import os
import glob
import numpy as np
import cv2
import csv

import torch
import torch.utils.data as data

from torchvision import transforms
from torchvision.utils import make_grid

import openslide as ops
import glymur

import matplotlib.pyplot as plt

from misc import utils
from config import Config



####
class SerializeFile(data.Dataset):
    """
    Read a single file as multiple patches of same shape- perform the padding beforehand
    """
    def __init__(self, file_path, window_size, mask_size):
        self.window_size = window_size
        self.patch_info_list = []

        file_ext = os.path.basename(file_path).split('.')[-1]
        if file_ext not in ['svs', 'ndpi']:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img, patch_info_list = self.__get_patch_list(img, window_size, mask_size)

        self.img = img
        self.patch_info_list = patch_info_list

    def __get_patch_list(self, img, window_size, mask_size):
        win_size = window_size
        msk_size = step_size = mask_size

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = img.shape[0]
        im_w = img.shape[1]

        last_h, _ = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, _ = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

        # generating subpatches index from orginal
        patch_info_list = []
        for iy, ridx in enumerate(list(range(0, last_h, step_size[0]))):
            for ix, cidx in enumerate(list(range(0, last_w, step_size[1]))):
                patch_info_list.append([[ridx, cidx], [iy, ix]])
        return img, patch_info_list

    def __read_region(self, coord, window_size):
        return self.img[coord[0]: coord[0] + window_size[0],
                        coord[1]: coord[1] + window_size[1]]

    def __getitem__(self, idx):

        coord, patch_idx = self.patch_info_list[idx]

        img_patch = self.__read_region(coord, self.window_size)

        # ! cant return list of integer, it will be turned into sthg else
        patch_idx = np.array(patch_idx, dtype=np.int32)
        return img_patch, patch_idx

    def __len__(self):
        return len(self.patch_info_list)
####


class SerializeWSI(data.Dataset):
    def __init__(self, wsi_path, wsi_ext, tile_info, window_size, mask_size, ds_factor, magnification_level=0, tissue=None):
        """
        Don't do padding for the WSI because bottom and left regions are likely just background       
        """
        self.wsi_ext = wsi_ext
        self.magnification_level = magnification_level
        self.ds_factor = ds_factor
        self.window_size = (window_size[0],window_size[1])
        self.mask_size = (mask_size[0],mask_size[1])
        self.patch_info_list = []

        if wsi_ext == 'jp2':
            self.wsi_open = glymur.Jp2k(wsi_path)
            fullres = self.wsi_open[:]
            wsi_shape = [fullres.shape[1], fullres.shape[0]]
        else:
            self.wsi_open = ops.OpenSlide(wsi_path)
            wsi_shape = [self.wsi_open.dimensions[1], self.wsi_open.dimensions[0]]

        if tissue is not None:
            self.ds_factor_tiss = int(round(wsi_shape[0] / tissue.shape[0])) 

        # get tile coordinate information and tile dimensions
        start_x = tile_info[0]
        start_y = tile_info[1]
        tile_shape_x = tile_info[2]
        tile_shape_y = tile_info[3]

        # ! only extract the portion within the original size
        for iy, ridx in enumerate(list(range(start_y, tile_shape_y, mask_size[0]))):
            for ix, cidx in enumerate(list(range(start_x, tile_shape_x, mask_size[1]))):
                # if tissue mask is provided, then only extract patches from valid tissue regions
                if tissue is not None:
                    win_tiss = tissue[
                        int(round(ridx / self.ds_factor_tiss)):int(round(ridx / self.ds_factor_tiss)) + int(
                            round(window_size[0] / self.ds_factor_tiss)),
                        int(round(cidx / self.ds_factor_tiss)):int(round(cidx / self.ds_factor_tiss)) + int(
                            round(window_size[1] / self.ds_factor_tiss))]
                    if np.sum(win_tiss) > 0:
                        self.patch_info_list.append([[cidx, ridx], [iy, ix]])
                else:
                    self.patch_info_list.append([[cidx, ridx], [iy, ix]])

    def __read_region(self, coord, window_size):
        if self.wsi_ext == 'jp2':
            patch = self.wsi_open[coord[1]:coord[1] + window_size[1] * self.ds_factor:self.ds_factor,
                                  coord[0]:coord[0] + window_size[0] * self.ds_factor:self.ds_factor, :]
        else:
            patch = self.wsi_open.read_region(coord, self.magnification_level, self.window_size)
            r, g, b, _ = cv2.split(np.array(patch))
            return cv2.merge([r, g, b])

    def __getitem__(self, idx):
        coord, patch_idx = self.patch_info_list[idx]
        img_patch = self.__read_region(coord, self.window_size)

        # ! cant return list of integer, it will be turned into sthg else
        patch_idx = np.array(patch_idx, dtype=np.int32)
        return img_patch, patch_idx

    def __len__(self):
        return len(self.patch_info_list)

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
