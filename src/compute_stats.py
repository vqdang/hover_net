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
valid_same_name = [
    'TCGA-21-5786-01Z-00-DX1',
    'TCGA-49-4488-01Z-00-DX1',
    'TCGA-A7-A13F-01Z-00-DX1',
    'TCGA-B0-5698-01Z-00-DX1',
    'TCGA-B0-5710-01Z-00-DX1',
    'TCGA-CH-5767-01Z-00-DX1',
    'TCGA-E2-A1B5-01Z-00-DX1',
    'TCGA-G9-6336-01Z-00-DX1',]

valid_diff_name = [
    'TCGA-AY-A8YK-01A-01-TS1',
    'TCGA-DK-A2I6-01A-01-TS1',
    'TCGA-G2-A2EK-01A-02-TSB',
    'TCGA-KB-A93J-01A-01-TS1',
    'TCGA-NH-A8F7-01A-01-TS1',
    'TCGA-RD-A8N9-01A-01-TS1',]

# true_dir = '../../data/NUC_Kumar/train-set/msks_fixed/imgs_all/'
# true_dir = '../../data/NUC_TNBC/Labels/'
# pred_dir = 'output/v6.0.1.0/TNBC/proc_tune/'

true_dir = '../../../data/NUC_Kumar/train-set/msks_fixed/imgs_all/'
pred_dir = 'output/XXXX_proc_e1m1/'

file_list = glob.glob(pred_dir + '*.png')
file_list.sort() # ensure same order [1]

metrics = [[], [], [], [], []]
for filename in file_list:
    filename = os.path.basename(filename)
    basename = filename.split('.')[0]
    
    if basename not in valid_same_name and \
        basename not in valid_diff_name:
        continue
    # if basename not in valid_same_name:
    #     continue
    
    # true = sio.loadmat(true_dir + basename + '.mat')
    # true = (true['result']).astype('int32')

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
