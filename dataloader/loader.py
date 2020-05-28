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
import albumentations as A

from misc.utils import cropping_center, center_pad_to_shape
from .augs import gen_instance_hv_map


class TrainSerialLoader(torch.utils.data.Dataset):
    """
    Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
    """
    def __init__(self, file_list, input_shape=None, mask_shape=None, mode='train'):
        assert input_shape is not None and mask_shape is not None
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.augment_img_mask, self.augment_img = self.__augmentation__(mode)
        self.info_list = file_list

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        img = (data[..., :3]).astype('uint8')  # RGB images
        ann = data[..., 3:]  # instance ID map and type map

        # perform augmentation
        aug_img_mask = self.augment_img_mask(image=img, mask=ann)
        img = aug_img_mask['image']
        ann = aug_img_mask['mask']

        aug_img = self.augment_img(image=img)
        img = aug_img['image']

        feed_dict = {'img': img}
        # * Specific on-the-fly processing for annotation label
        if ann.shape[-1] == 2:  # nuclei segmentation + classification
            inst_map, type_map = np.dsplit(ann, -1)
            np_map = np.array(inst_map > 0, dtype='uint8')
            hv_map = gen_instance_hv_map(inst_map, self.mask_shape)
            # binarise instance map, ordering of operaton matters !
            np_map = cropping_center(np_map, self.mask_shape)
            hv_map = cropping_center(hv_map, self.mask_shape)
            tp_map = cropping_center(type_map, self.mask_shape)
            feed_dict['np_map'] = np_map[..., None]  # HWC
            feed_dict['hv_map'] = hv_map
            feed_dict['tp_map'] = tp_map[..., None]  # HWC
        else:  # nuclei segmentation only
            inst_map = ann[..., 0]  # HW1 -> HW
            np_map = np.array(inst_map > 0, dtype='uint8')
            hv_map = gen_instance_hv_map(inst_map, self.mask_shape)
            np_map = cropping_center(np_map, self.mask_shape)
            hv_map = cropping_center(hv_map, self.mask_shape)
            feed_dict['np_map'] = np_map[..., None]  # HWC
            feed_dict['hv_map'] = hv_map

        return feed_dict

    @staticmethod
    def view(batch_data):
        def prep_sample(data):
            shape_array = [np.array(v.shape[:2]) for v in data.values()]
            shape = np.maximum(*shape_array)

            cmap = plt.get_cmap('jet')

            def colorize(ch, vmin, vmax):
                ch = np.squeeze(ch.astype('float32'))
                ch = ch / (vmax - vmin + 1.0e-16)
                # take RGB from RGBA heat map
                ch_cmap = (cmap(ch)[..., :3] * 255).astype('uint8')
                ch_cmap = center_pad_to_shape(ch_cmap, shape)
                return ch_cmap

            viz_list = []
            # cmap may randomly fails if of other types
            viz_list.append(colorize(data['np_map'], 0, 1))
            # map to [0,2] for better visualisation.
            # Note, [-1,1] is used for training.
            viz_list.append(colorize(data['hv_map'][..., 0] + 1, 0, 2))
            viz_list.append(colorize(data['hv_map'][..., 1] + 1, 0, 2))
            img = center_pad_to_shape(data['img'], shape)
            prepped_img = np.concatenate([img] + viz_list, axis=1)
            return prepped_img

        batch_size = list(batch_data.values())[0].shape[0]
        for sample_idx in range(batch_size):
            sample_data = {k: v[sample_idx] for k, v in batch_data.items()}
            displayed_sample = prep_sample(sample_data)
            plt.subplot(batch_size, 1, sample_idx+1)
            plt.imshow(displayed_sample)
        plt.show()
        return

    def __augmentation__(self, mode):
        """
        Augmentation pipeline. For more information on how this can be modified, refer to:
        https://albumentations.readthedocs.io/en/latest/api/augmentations.html#module-albumentations.augmentations.transforms
        """
        if mode == 'train':
            augment_img_mask = A.Compose([
                A.ShiftScaleRotate(shift_limit=5, scale_limit=0.2,
                                   rotate_limit=179, interpolation=cv2.INTER_NEAREST),
                A.VerticalFlip(p=.5),
                A.HorizontalFlip(p=.5),
                A.ElasticTransform(alpha_affine=0, alpha=35,
                                   sigma=5, interpolation=cv2.INTER_NEAREST),
                A.CenterCrop(self.input_shape[0], self.input_shape[1])
            ])

            augment_img = A.Compose([
                # different ordering will give different results (not commutative)
                A.OneOf([
                    A.Compose([
                        A.HueSaturationValue(hue_shift_limit=(-25, 0),
                                             sat_shift_limit=0, val_shift_limit=0, p=1),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3, contrast_limit=0.4, p=1)
                    ]),
                    A.Compose([
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3, contrast_limit=0.4, p=1),
                        A.HueSaturationValue(hue_shift_limit=(-25, 0),
                                             sat_shift_limit=0, val_shift_limit=0, p=1)
                    ])
                ]),
                A.OneOf([A.MedianBlur(blur_limit=3),
                         A.GaussianBlur(blur_limit=3),
                         A.GaussNoise(var_limit=0.05*255)
                         ])
            ])

        else:
            augment_img_mask = A.Compose([
                A.CenterCrop(self.input_shape[0], self.input_shape[1])
            ])
            augment_img = A.NoOp()

        return augment_img_mask, augment_img
