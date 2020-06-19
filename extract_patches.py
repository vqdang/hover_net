"""extract_patches.py

Patch extraction script.
"""


import glob
import os
import tqdm

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

#-------------------------------------------------------------------------------------
if __name__ == '__main__':

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True 

    save_root = ""
    win_size = [540, 540]    # Size of patch to extract. Should be at least twice time larger than
                            # self.train_base_shape(config.py) to reduce the padding effect during augmentation. 
    step_size = [80, 80]     # Step size for patch extraction. Should match network output size. 
    dataset_name = 'consep'   # Name of dataset - use Kumar, CPM17 or CoNSeP. Pulls dataset info from dataset.py
    extract_type = 'mirror'  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    xtractor = PatchExtractor(win_size, step_size)

    dataset_info = get_dataset(dataset_name)
    for dir_codename, dir_desc in dataset_info.desc.items():
        img_ext, img_dir = dir_desc['img']
        ann_ext, ann_dir = dir_desc['ann']

        out_dir = "%s/train/%s/%s/%dx%d_%dx%d/" % \
                    (save_root, dataset_name, dir_codename,
                    win_size[0], win_size[1], step_size[0], step_size[1])
        file_list = glob.glob('%s/*%s' % (img_dir, img_ext))
        file_list.sort() # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = 'Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]'
        pbarx = tqdm.tqdm(total=len(file_list), bar_format=pbar_format, ascii=True, position=0)

        for file_idx, filename in enumerate(file_list):
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]

            img = dataset_info.load_img(img_dir + basename + img_ext)
            ann = dataset_info.load_ann(ann_dir + basename + ann_ext, type_classification)

            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = 'Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm.tqdm(total=len(sub_patches), leave=False, bar_format=pbar_format, ascii=True, position=1)

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)
                pbar.update()
            pbar.close()

            pbarx.update()
        pbarx.close()
