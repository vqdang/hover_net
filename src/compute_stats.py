import argparse
import cProfile as profile
import glob
import os

import cv2
import numpy as np
import scipy.io as sio
import pandas as pd

from metrics.stats_utils import *

def run_nuclei_type_stat(pred_dir, true_dir, type_uid_list=None, exhaustive=True):
    """
    GT must be exhaustively annotated for instance location (detection)

    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image. 
                            Each .mat must contain:

                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index

                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance

        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.

        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types
    """
    ###
    file_list = glob.glob(pred_dir + '*.mat')
    file_list.sort() # ensure same order [1]

    paired_all = [] # unique matched index pair
    unpaired_true_all = [] # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = [] # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = [] # each index is 1 independent data point
    pred_inst_type_all = [] # each index is 1 independent data point
    for file_idx, filename in enumerate(file_list[:]):
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]

        true_info = sio.loadmat(true_dir + basename + '.mat')
        # dont squeeze, may be 1 instance exist
        true_centroid  = (true_info['inst_centroid']).astype('float32')
        true_inst_type = (true_info['inst_type']).astype('int32')

        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:,0]                
        else: # no instance at all
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        # * for converting the GT type in CoNSeP
        # true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 3
        # true_inst_type[(true_inst_type == 5) | (true_inst_type == 6) | (true_inst_type == 7)] = 4

        pred_info = sio.loadmat(pred_dir + basename + '.mat')
        # dont squeeze, may be 1 instance exist
        pred_centroid  = (pred_info['inst_centroid']).astype('float32')
        pred_inst_type = (pred_info['inst_type']).astype('int32')

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:,0]
        else: # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(true_centroid, pred_centroid, 12)
 
        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        pred_idx_offset = pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0: # ! sanity
            paired[:,0] += true_idx_offset
            paired[:,1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    paired_true_type = true_inst_type_all[paired_all[:,0]]
    paired_pred_type = pred_inst_type_all[paired_all[:,1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)
        
        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum() 
        fn_d = (unpaired_true == type_id).sum() 

        f1_type = (2 * (tp_dt + tn_dt)) / \
                    (2 * (tp_dt + tn_dt) + w[0] * fp_dt + w[1] * fn_dt \
                                         + w[2] *  fp_d + w[3] * fn_d)
        return f1_type

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0] 
    fn_d = unpaired_true_type.shape[0] 

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    w = [2, 2, 1, 1]
    
    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

    results_list = [f1_d, acc_type]
    for type_uid in type_uid_list:
        f1_type = _f1_type(paired_true_type, paired_pred_type, 
                        unpaired_pred_type, unpaired_true_type, type_uid, w)
        results_list.append(f1_type)

    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    print(np.array(results_list))
    return

def run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False, ext='.mat'):
    # print stats of each image
    print(pred_dir)

    file_list = glob.glob('%s/*%s' % (pred_dir, ext))
    file_list.sort() # ensure same order

    metrics = [[], [], [], [], [], []]
    for filename in file_list[:]:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]

        true = sio.loadmat(true_dir + basename + '.mat')
        true = (true['inst_map']).astype('int32')

        pred = sio.loadmat(pred_dir + basename + '.mat')
        pred = (pred['inst_map']).astype('int32')

        # to ensure that the instance numbering is contiguous
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(pq_info[0]) # dq
        metrics[2].append(pq_info[1]) # sq
        metrics[3].append(pq_info[2]) # pq
        metrics[4].append(get_fast_aji_plus(true, pred))
        metrics[5].append(get_fast_aji(true, pred))

        if print_img_stats:
            print(basename, end="\t")
            for scores in metrics:
                print("%f " % scores[-1], end="  ")
            print()
    ####
    metrics = np.array(metrics)
    metrics_avg = np.mean(metrics, axis=-1)
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    print(metrics_avg)
    metrics_avg = list(metrics_avg)
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help="mode to run the measurement,"
                                "`type` for nuclei instance type classification or"
                                "`instance` for nuclei instance segmentation",
                        nargs='?', default='instance', const='instance')
    parser.add_argument('--pred_dir', help="point to output dir", nargs='?', default='', const='')
    parser.add_argument('--true_dir', help="point to ground truth dir", nargs='?', default='', const='')
    args = parser.parse_args()

    if args.mode == 'instance':
        run_nuclei_inst_stat(args.pred_dir, args.true_dir, print_img_stats=False)
    if args.mode == 'type':
        run_nuclei_type_stat(args.pred_dir, args.true_dir)

