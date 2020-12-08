import argparse
import cProfile as profile
import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

from metrics.stats_utils import (get_dice_1, get_fast_aji, get_fast_aji_plus,
                                 get_fast_pq, pair_coordinates)


####
def run_nuclei_type_stat(
        pred_df_list, # centroid, type, df frame 
        true_df_list, # column should match between pred and true
        type_uid_list=None, exhaustive=True):
    """
    GT must be exhaustively annotated for instance location (detection)

        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.

        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types
    """
    ###
    assert len(pred_df_list) == len(true_df_list)
    nr_img = len(pred_df_list)

    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point

    for idx in range(nr_img):
        # dont squeeze, may be 1 instance exist
        true_df = true_df_list[idx]
        true_centroid  = true_df[['x', 'y']].to_numpy().astype('int32')
        true_inst_type = true_df['type'].to_numpy().astype('int32')

        if true_centroid.shape[0] == 0: # no instance at all
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        pred_df = pred_df_list[idx]
        pred_centroid  = pred_df[['x', 'y']].to_numpy().astype('int32')
        pred_inst_type = pred_df['type'].to_numpy().astype('int32')

        if pred_centroid.shape[0] == 0: # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, 12)

        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = true_idx_offset + \
            true_inst_type_all[-1].shape[0] if idx != 0 else 0
        pred_idx_offset = pred_idx_offset + \
            pred_inst_type_all[-1].shape[0] if idx != 0 else 0
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
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
    # print(paired_all.shape, paired_all.dtype)
    # print(unpaired_true_all.shape, unpaired_true_all.dtype)
    # print(unpaired_pred_all.shape, unpaired_pred_all.dtype)
    # print(true_inst_type_all.shape, true_inst_type_all.dtype)
    # print(pred_inst_type_all.shape, pred_inst_type_all.dtype)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
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
            (2 * (tp_dt + tn_dt) + w[0] * fp_dt + w[1] * fn_dt
             + w[2] * fp_d + w[3] * fn_d)
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


####
def run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False, ext='.mat'):
    # print stats of each image
    print(pred_dir)

    file_list = glob.glob('%s/*%s' % (pred_dir, ext))
    file_list.sort()  # ensure same order

    metrics = [[], [], [], [], [], []]
    for filename in file_list[:]:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]

        true = np.load(true_dir + basename + '.npy')
        true = true.astype('int32')
        # true = true[..., 0]  # HxWx1, uncomment to use label class from CoNSeP

        pred = sio.loadmat(pred_dir + basename + '.mat')
        pred = (pred['inst_map']).astype('int32')

        # to ensure that the instance numbering is contiguous
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(pq_info[0])  # dq
        metrics[2].append(pq_info[1])  # sq
        metrics[3].append(pq_info[2])  # pq
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
    import glob
    import os
    import pandas as pd
    import pathlib
    from misc.utils import check_available_subject


    def map_pred_type(df):
        df = df.copy()
        df.insert(3, 'type', df['name'])
        type_info_dict = {
            'nolabe' : 0, 
            'neopla' : 1, 
            'inflam' : 2, 
            'connec' : 3, 
            'necros' : 4, 
            'no-neo' : 5, 
        }
        for k, v in type_info_dict.items():
            df.loc[df['name'] == k, 'type'] = v
        return df
    def map_true_type(df):
        df = df.copy()
        df.insert(3, 'type', df['name'])
        return df

    true_dir = 'dataset/rmt/continual_v0.0/anns_katharina_fix/'

    name_list = [
        'BR3804063_Scan1@007112-003616@008112-004616',
        'CH3804018_Scan1@018149-019387@018757-020004',
        'CH3804020_Scan1@011708-017926@012316-018543',
        'CH3804031_Scan1@008580-010101@009188-010718',
        'CH3804077_Scan1@008538-014007@009146-014624',

        # 'ES3804019_Scan2@011411-016061@012019-016678',
        # 'NO3804023_Scan1@003696-003172@004696-004172',
        # 'WP3804008_Scan1@008129-018279@009129-019279',
        # 'WP3804008_Scan1@008722-011217@009722-012217',
        # 'WP3804020_Scan1@009703-010517@010703-011517',
    ]

    pred_dir_list = [
        'dataset/rmt/continual_v0.0/pred/base/',
        'dataset/rmt/continual_v0.0/pred/continual_data=[v0.0]_run=[v0.0]/',
        'dataset/rmt/continual_v0.0/pred/continual_data=[v0.0]_run=[v0.1]/',
        'dataset/rmt/continual_v0.0/pred/clf=[densenet-mini_v0.2]/',
        'dataset/rmt/continual_v0.0/pred/clf=[densenet-mini_v0.2.1]/',
        'dataset/rmt/continual_v0.0/pred/clf=[densenet-mini_v0.2.2]/',
    ]

    # true_file_list = glob.glob('%s/*.tsv' % true_dir)
    # pred_file_list = glob.glob('%s/*.tsv' % pred_dir)
    # true_file_list = [pathlib.Path(v).stem.replace('.png-points','') for v in true_file_list]
    # pred_file_list = [pathlib.Path(v).stem.replace('_nuclei_dict','') for v in pred_file_list]
    # _, _, name_list = check_available_subject(true_file_list, pred_file_list) 

    # [ 0.95638  0.92664  0.80346  0.80888  0.90647]
    for pred_dir in pred_dir_list:
        print(pred_dir)
        true_df_list = ['%s/%s.png-points.tsv' % (true_dir, v) for v in name_list]    
        pred_df_list = ['%s/%s.tsv' % (pred_dir, v) for v in name_list]
        true_df_list = [pd.read_csv(v, sep='\t') for v in true_df_list]   
        pred_df_list = [pd.read_csv(v, sep='\t') for v in pred_df_list]   
        true_df_list = [map_true_type(v) for v in true_df_list]   
        pred_df_list = [map_pred_type(v) for v in pred_df_list]   
        run_nuclei_type_stat(pred_df_list, true_df_list)

    print('here')
