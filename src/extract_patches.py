
import glob
import os

import cv2
import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

###########################################################################
if __name__ == '__main__':

    extract_type = 'mirror' # 'valid' for fcn8 segnet etc.
                            # 'mirror' for u-net etc.
    # check the patch_extractor.py 'main' to see the different

    step_size = [ 80,  80] # should match self.train_mask_shape (config.py) 
    win_size  = [600, 600] # should be at least twice time larger than 
                           # self.train_base_shape (config.py) to reduce 
                           # the padding effect during augmentation

    xtractor = PatchExtractor(win_size, step_size)

    ####
    data_mode = 'valid_diff'
    # 5784 is the stain code in SN-SN
    stain_code = 'XXXX' # 5784 A1AS A2I6 A13E XXXX
    img_dir = '../../data/NUC_Kumar/train-set/imgs_norm/%s/' % stain_code
    ann_dir = '../../data/NUC_Kumar/train-set/anns_proc/imgs_%s/' % data_mode
    img_ext = 'tif'
    ####
    out_root_path = "/media/Train/Kumar/"
    out_dir = "%s/paper/%s/%s/" % (out_root_path, data_mode, stain_code)
    
    file_list = glob.glob(ann_dir + '*.npy')
    file_list.sort() 

    rm_n_mkdir(out_dir)

    for filename in file_list:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]

        img = cv2.imread(img_dir + basename + img_ext)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        ann = np.load(ann_dir + basename + '.npy')

        img = np.concatenate([img, ann], axis=-1)
        sub_patches = xtractor.extract(img, extract_type)
        for idx, patch in enumerate(sub_patches):
            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)
