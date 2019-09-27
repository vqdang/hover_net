
import glob
import os

import cv2
import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from config import Config

###########################################################################
if __name__ == '__main__':
    
    cfg = Config()

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
        
        if cfg.type_classification:
            ann = np.load(ann_dir + basename + '.npy')
            ann_inst = ann[...,0]
            ann_type = ann[...,1]
            
            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6)] = 4
            
            ann = np.concatenate([ann_inst, ann_type], axis=-1)
            ann = ann.astype('int32')
            assert np.max(ann[...,1]) <= 4, np.max(ann[...,1])
        
        else:
            # assumes that ann is WxH; if WxHx2 (class labels available) then extract first channel after loading
            ann_inst = np.load(ann_dir + basename + '.npy')
            ann_inst = ann_inst.astype('int32')
            ann = np.expand_dims(ann_inst, -1)
       
        img = np.concatenate([img, ann], axis=-1)
        sub_patches = xtractor.extract(img, extract_type)
        for idx, patch in enumerate(sub_patches):
            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)
