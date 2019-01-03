import argparse
import colorsys
import glob
import math
import os
import random
import re  # regex
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib import cm
from scipy.ndimage import measurements

#####--------------------------Optimized for Speed
def get_fast_aji(true, pred, mode='plus'):
    """
    Fast AJI

    Args:
        mode: 'plus' to use the AJI+ version else, using the orignal
               AJI computation
    """
    true = np.copy(true)
    pred = np.copy(pred)

    if mode == 'plus':
        pred = remap_label(pred)
        true = remap_label(true)

    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred)) 

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    
    overall_inter = 0
    overall_union = 0
    avail_pred_idx = list(range(1, len(pred_id)))
    for true_idx in range(1, len(true_id)):
        max_iou   = 0
        max_inter = 0 
        max_union = 0
        max_iou_idx = -1

        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)

        overlap_idx = []
        for idx in pred_true_overlap_id:
            if idx in avail_pred_idx:
                overlap_idx.append(idx)

        for pred_idx in overlap_idx:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            union = total - inter
            iou = inter / union
            if iou > max_iou:
                max_iou = iou
                max_inter = inter
                max_union = union
                max_iou_idx = pred_idx

        overall_inter += max_inter
        if max_iou > 0:
            overall_union += max_union
            avail_pred_idx.remove(max_iou_idx)
        else: # total missed, so just add GT
            overall_union += t_mask.sum()            

    # deal with remaining i.e over segmented
    for pred_idx in avail_pred_idx:
        p_mask = pred_masks[pred_idx]
        overall_union += p_mask.sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score
#####
def get_fast_dice_2(true, pred):
    """
        Ensemble dice
    """
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    overall_total = 0
    overall_inter = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
 
    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try: # blinly remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just mean no background
        for pred_idx in pred_true_overlap_id:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / overall_total
#####
def get_fast_panoptic_quality(true, pred, match_iou=0.75):
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
 
    overall_iou = 0
    pred_true_match_dict = {}
    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_idx in pred_true_overlap_id:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            union = total - inter
            iou = inter / union
            if iou > match_iou:
                if pred_idx in pred_true_match_dict.keys():
                    print('Warning TRUE ID=%d: PRED ID=%d is already matched to TRUE ID=%d',
                            true_idx, pred_idx, pred_true_match_dict[pred_idx])
                else:
                    pred_true_match_dict[pred_idx] = true_idx
                    overall_iou += iou

    match_pred_list = pred_true_match_dict.keys()
    match_true_list = pred_true_match_dict.values()
    tp_instances = len(match_pred_list)
    fp_instances = len([idx for idx in pred_id if idx not in match_pred_list])
    fn_instances = len([idx for idx in true_id if idx not in match_true_list])
    # print('------', tp_instances, fp_instances, fn_instances)

    detection_quality = tp_instances / (tp_instances + 0.5 * fp_instances + 0.5 * fn_instances)
    segmentation_quality = overall_iou / (tp_instances + 1.0e-6) # epsilon for stability

    return detection_quality, segmentation_quality
#####--------------------------As pseudocode
def get_dice_1(true, pred):
    """
        Traditional dice
    """
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)
####
def get_dice_2(true, pred):
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    # remove background aka id 0
    true_id.remove(0)
    pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = np.array(true == t, np.uint8)
        for p in pred_id:
            p_mask = np.array(pred == p, np.uint8)
            intersect = p_mask * t_mask          
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += (t_mask.sum() + p_mask.sum())
    return 2 * total_intersect / total_markup
#####
def get_aji(true, pred):
    """
        Ported from MoNuSeg matlab aggregated_jaccard_index.m code, 
        original version, not AJI+.

        For demo purpose only, for actual calculation, see get_fast_aji
        with better optimization and bug fixed

        The aggregated_jaccard_index.m has a bug in the following portion 
        >    % Find best matched nuclei
        >    [maxji, indmax] = max(JI);
        >    predicted_nuclei = 1*(predicted_map == indmax);
        >    % update intersection and union pixel counts
        >    overall_correct_count = overall_correct_count + nnz(and(gt,predicted_nuclei));
        >    union_pixel_count = union_pixel_count + nnz(or(gt,predicted_nuclei));
        >    
        >    % remove used predicted nuclei from predicted map
        >    predicted_map = (1 - predicted_nuclei).*predicted_map;   

        In the case with no intersection between all predicted segments against current
        ground truth segment under consideration
        >    % Find best matched nuclei
        >    [maxji, indmax] = max(JI);
        >    predicted_nuclei = 1*(predicted_map == indmax);
        will default to first location in the the predicted segmented array 
        (index 1 in matlab and 0 in python)
        
        As a result, the nuclei with ID 1 will be accidentally included togther with the
        current groundtruth to the overall union pixel count (more penalty)
        >    union_pixel_count = union_pixel_count + nnz(or(gt,predicted_nuclei));        
        and also be removed from later calculation (prevent later correct intersection)
        >    predicted_map = (1 - predicted_nuclei).*predicted_map;   
    """
    warnings.warn('\nWARNING-Contain Same Bug as Original Code !!! Read the code for detail!')

    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    # remove background aka id 0
    true_id.remove(0)
    pred_id.remove(0)

    overall_inter = 0
    overall_union = 0
    for true_idx in range(0, len(true_id)):
        # Compute JI of each gt with every predicted nuclei
        iou_list = np.zeros(len(pred_id), np.float32)
        t_mask = np.array(true == true_id[true_idx], np.uint8)
        for pred_idx in range(0, len(pred_id)):
            # extract j-th predicted nuclei
            p_mask = np.array(pred == pred_id[pred_idx], np.uint8)
            inter = (p_mask * t_mask).sum()
            total = (p_mask + t_mask).sum()
            # compute ratio of cardinalities of intersection and union pixels 
            iou_list[pred_idx] = inter / (total - inter)

        # Find best matched nuclei
        max_pred_idx = pred_id[np.argmax(iou_list)]
        max_pred_mask = np.array(pred == max_pred_idx, np.uint8)
        max_inter = (max_pred_mask * t_mask).sum()
        max_total = (max_pred_mask + t_mask).sum()
        max_union = max_total - max_inter
        # update intersection and union pixel counts
        overall_inter += max_inter
        overall_union += max_union

        # remove used predicted nuclei from predicted map
        pred[pred == max_pred_idx] = 0
    # add all unmatched pixels left in the predicted map to union set
    overall_union += np.array(pred > 0, np.uint8).sum()
    aji = overall_inter / overall_union
    return aji
#####
def remap_label(pred, by_size=True):
    """
    Rename all instance id so that the id (sorting from min to max)
    is contiguos i.e [0, 1, 2, 3] not [0, 2, 4, 6].

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, id in enumerate(pred_id):
        new_pred[pred == id] = idx + 1    
    return new_pred
####