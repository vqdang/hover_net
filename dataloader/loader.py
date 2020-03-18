import csv
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data

from imgaug import augmenters as iaa

from misc.utils import cropping_center, center_pad_to_shape
from .augs import gen_instance_hv_map

class TrainSerialLoader(torch.utils.data.Dataset):
    def __init__(self, file_list, input_shape=None, mask_shape=None, mode='train'):
        assert input_shape is not None and mask_shape is not None
        self.mask_shape = mask_shape
        self.input_shape = input_shape

        self.info_list = file_list
        augmentor = self.__augmentation__(mode)
        self.shape_augs = iaa.Sequential(augmentor[0]) 
        self.input_augs = iaa.Sequential(augmentor[1]) 

    def __len__(self):
        return len(self.info_list)        

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        img = (data[...,:3]).astype('uint8') # RGB images
        ann = data[...,3:] # instance ID map and type map

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        # * Specific on the flight processing for annotation label
        if ann.shape[-1] == 2: # Nuclei Segmentation + Type Classification
            inst_map, type_map = np.dsplit(ann, -1)
            hv_map = gen_instance_hv_map(inst_map, self.mask_shape)
            # binarize instance map
            inst_map[inst_map > 0] = 1
        else: # Nuclei Segmentation only
            inst_map = ann[...,0] # HW1 -> HW
            hv_map = gen_instance_hv_map(inst_map, self.mask_shape)
            # binarize instance map
            inst_map[inst_map > 0] = 1
        ann = cropping_center(ann, self.mask_shape)

        return img, ann

    @staticmethod
    def view(img_list, ann_list):
        def prep_imgs(img, ann):
            shape = np.maximum(img.shape[:2], ann.shape[:2])

            cmap = plt.get_cmap('viridis')
            # cmap may randomly fails if of other types
            ann = ann.astype('float32')
            ann_chs = np.dsplit(ann, ann.shape[-1])
            for i, ch in enumerate(ann_chs):
                ch = np.squeeze(ch)
                # normalize to -1 to 1 range else
                # cmap may behave stupidly
                ch = ch / (np.max(ch) - np.min(ch) + 1.0e-16)
                # take RGB from RGBA heat map
                ann_ch_cmap = cmap(ch)[...,:3] 
                ann_ch_cmap = center_pad_to_shape(ann_ch_cmap, shape)
                ann_chs[i]  = ann_ch_cmap
            img = img.astype('float32') / 255.0
            img = center_pad_to_shape(img, shape)
            prepped_img = np.concatenate([img] + ann_chs, axis=1)
            return prepped_img

        for idx in range (len(img_list)):
            displayed_img = prep_imgs(img_list[idx], ann_list[idx])
            plt.subplot(len(img_list), 1, idx+1)
            plt.imshow(displayed_img)
        plt.show()        
        return

    def __augmentation__(self, mode):
        if mode == 'train':
            shape_augs = [
                iaa.PadToFixedSize(
                    1800, 1800, 
                    pad_cval=255,
                    position='center',
                    deterministic=True, 
                ),
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), 
                           "y": (0.8, 1.2)}, 
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), 
                                       "y": (-0.01, 0.01)}, 
                    shear=(-5, 5), # shear by -5 to +5 degrees
                    rotate=(-179, 179), # rotate by -179 to +179 degrees
                    order=0,    # use nearest neighbour
                    backend='cv2' # opencv for fast processing
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(self.input_shape[0], 
                                    self.input_shape[1],
                                    position='center'
                ),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ]
        
            input_augs = [
                iaa.OneOf([
                            iaa.GaussianBlur((0, 2.0)), # gaussian blur with random sigma
                            iaa.MedianBlur(k=(1, 3)), # median with random kernel sizes
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                            ]),
                iaa.Sequential([
                    iaa.Add((-26, 26)),
                    iaa.AddToHueAndSaturation((-20, 20)),
                    iaa.LinearContrast((0.75, 1.25), per_channel=1.0),
                ], random_order=True),
            ]   
        elif mode == 'valid':
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(self.input_shape[0], 
                                    self.input_shape[1],
                                    position='center')
            ]
            input_augs = [
            ]

        return shape_augs, input_augs