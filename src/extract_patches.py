
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

    # orignal size (win size) - input size - output size (step size)
    # 512x512 - 256x256 - 256x256 fcn8, dcan, segnet
    # 536x536 - 268x268 - 84x84   unet, dist
    # 540x540 - 270x270 - 80x80   xy, hover
    # 504x504 - 252x252 - 252x252 micronet
    step_size = [252, 252] # should match self.train_mask_shape (config.py) 
    win_size  = [504, 504] # should be at least twice time larger than 
                           # self.train_base_shape (config.py) to reduce 
                           # the padding effect during augmentation

    xtractor = PatchExtractor(win_size, step_size)

    ###
    img_ext = '.png'
    data_mode = 'valid'
    img_dir = '../../../data/NUC_UHCW/No_SN/%s/' % data_mode 
    ann_dir = '../../../data/NUC_UHCW/Labels_class/' 
    ####
    out_root_path = "../../../train/UHCW/%dx%d_%dx%d" % \
                        (win_size[0], win_size[1], step_size[0], step_size[1])
    out_dir = "%s/%s/%s/" % (out_root_path, data_mode, 'XXXX')

    file_list = glob.glob('%s/*%s' % (img_dir, img_ext))
    file_list.sort() 

    rm_n_mkdir(out_dir)
    for filename in file_list:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]
        print(filename)

        img = cv2.imread(img_dir + basename + img_ext)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        ann = np.load(ann_dir + basename + '.npy')
        ann = ann.astype('int32')

        # merge class for CoNSeP
        ann_type = ann[...,1]
        ann_type[(ann_type == 3) | (ann_type == 4)] = 3
        ann_type[(ann_type == 5) | (ann_type == 6)] = 4
        assert np.max(ann[...,1]) <= 4, np.max(ann[...,1])

        img = np.concatenate([img, ann], axis=-1)
        sub_patches = xtractor.extract(img, extract_type)
        for idx, patch in enumerate(sub_patches):
            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)
