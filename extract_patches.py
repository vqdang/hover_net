
import glob
import os

import cv2
import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from config import Config

import importlib
from dataloader import dataset

###########################################################################
if __name__ == '__main__':
    
    type_classification = False

    extract_type = 'mirror' # 'valid' for fcn8 segnet etc.
                            # 'mirror' for u-net etc.
    # check the patch_extractor.py 'main' to see the different

    # orignal size (win size) - input size - output size (step size)
    # 512x512 - 256x256 - 256x256 fcn8, dcan, segnet
    # 536x536 - 268x268 - 84x84   unet, dist
    # 540x540 - 270x270 - 80x80   xy, hover
    # 504x504 - 252x252 - 252x252 micronet
    step_size = [80, 80] # should match self.train_mask_shape (config.py) 
    win_size  = [540, 540] # should be at least twice time larger than 
                           # self.train_base_shape (config.py) to reduce 
                           # the padding effect during augmentation

    xtractor = PatchExtractor(win_size, step_size)

    # TODO: centralized dumping output dir or manual insertion by hand
    output_root_dirt = 'dataset/train/'

    dataset_name = 'Kumar'
    dataset_info = getattr(dataset, dataset_name)()

    for dir_codename, dir_desc in dataset_info.desc.items():
        img_ext, img_dir = dir_desc['img']
        ann_ext, ann_dir = dir_desc['ann']

        # TODO: factor this out
        out_dir = "%s/%s/%s/%dx%d_%dx%d" % \
                        (output_root_dirt, dataset_name, dir_codename,
                        win_size[0], win_size[1], step_size[0], step_size[1])
        print(img_dir)
        file_list = glob.glob('%s/*%s' % (img_dir, img_ext))
        file_list.sort() 

        rm_n_mkdir(out_dir)
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]
            print(filename)

            # TODO: factor extension
            img = dataset_info.load_img(img_dir + basename + img_ext)
            ann = dataset_info.load_ann(ann_dir + basename + ann_ext)
            # print(img.shape, ann.shape) 

            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)
            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)
