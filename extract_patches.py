"""extract_patches.py

Patch extraction script.
"""


import glob
import os
import sys

import cv2
import numpy as np
import scipy.io as sio
import importlib

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

import dataset
from config import Config


cfg = Config()
type_classification = cfg.type_classification  # Determines whether to extract type map (only applicable to datasets with class labels).

win_size = [540, 540]    # Size of patch to extract. Should be at least twice time larger than
                         # self.train_base_shape(config.py) to reduce the padding effect during augmentation. 
step_size = [80, 80]     # Step size for patch extraction. Should match network output size. 
dataset_name = 'Kumar'   # Name of dataset - use Kumar, CPM17 or CoNSeP. Pulls dataset info from dataset.py
extract_type = 'mirror'  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.


if __name__ == '__main__':

    if type_classification and (dataset_name == 'Kumar' or dataset_name == 'CPM17'):
        raise Exception(
            'The dataset supplied must contain pixel-level class labels.')

    xtractor = PatchExtractor(win_size, step_size)

    dataset_info = getattr(dataset, dataset_name)()
    for dir_codename, dir_desc in dataset_info.desc.items():
        img_ext, img_dir = dir_desc['img']
        ann_ext, ann_dir = dir_desc['ann']

        out_dir = "%s/%s/patches/%s/" % \
            (dataset_info.data_root, dataset_name, dir_codename)
        print('Root dir:', img_dir)
        file_list = glob.glob('%s/*%s' % (img_dir, img_ext))
        file_list.sort()

        rm_n_mkdir(out_dir)
        for idx, filename in enumerate(file_list):
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]

            sys.stdout.write("\rExtracting patches from %s (%d/%d)" % (
                basename, idx + 1, len(file_list)))
            sys.stdout.flush()

            img = dataset_info.load_img(img_dir + basename + img_ext)
            ann = dataset_info.load_ann(ann_dir + basename + ann_ext)

            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)
            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir,
                                                     basename, idx), patch)
