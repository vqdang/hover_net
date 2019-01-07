import os
import glob
import numpy as np
import scipy.io as sio

from metrics.stats_utils import (remap_label,
                                get_dice_1, get_fast_aji,
                                get_fast_panoptic_quality)

####

# print stats of each image
print_img_stats = False

####

# directory contains the ground truth
true_dir = '' 
# directory contains the processed output
pred_dir = '' 

file_list = glob.glob(pred_dir + '*.png')
file_list.sort() # ensure same order [1]

metrics = [[], [], [], [], []]
for filename in file_list:
    filename = os.path.basename(filename)
    basename = filename.split('.')[0]
    
    true = np.load(true_dir + basename + '.npy')
    true = true.astype('int32')

    pred = sio.loadmat(pred_dir + basename + '_predicted_map.mat')
    pred = (pred['predicted_map']).astype('int32')

    pred = remap_label(pred, by_size=True)
    true = remap_label(true, by_size=True)
    panoptic_quality = get_fast_panoptic_quality(true, pred, match_iou=0.5)
    metrics[0].append(get_dice_1(true, pred))
    metrics[1].append(panoptic_quality[0]) # dq
    metrics[2].append(panoptic_quality[1]) # sq
    metrics[3].append(metrics[1][-1] * metrics[2][-1])
    metrics[4].append(get_fast_aji(true, pred))
        
    print(basename, end="\t")
    if print_img_stats:
        for scores in metrics:
            print("%f " % scores[-1], end="\t")
    print()
####
metrics = np.array(metrics)
metrics = np.mean(metrics, axis=-1)
metrics = list(metrics)
print(np.array(metrics))
