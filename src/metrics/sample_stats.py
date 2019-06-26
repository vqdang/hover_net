
import argparse
import math
import os
import random
import re  # regex

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib import cm

from stats_utils import *

#####
def compute_stat():
    true   = cv2.imread('sample/true.png', cv2.IMREAD_GRAYSCALE)
    pred_1 = cv2.imread('sample/pred_1.png', cv2.IMREAD_GRAYSCALE)
    pred_2 = cv2.imread('sample/pred_2.png', cv2.IMREAD_GRAYSCALE)

    true   = remap_label(true, by_size=False)
    pred_1 = remap_label(pred_1, by_size=False)
    pred_2 = remap_label(pred_2, by_size=False)

    pair_1_dice_2 = get_fast_dice_2(true, pred_1)
    pair_2_dice_2 = get_fast_dice_2(true, pred_2)
    pair_1_ajip = get_fast_aji_plus(true, pred_1)
    pair_2_ajip = get_fast_aji_plus(true, pred_2)
    pair_1_ajis = get_fast_aji(true, pred_1)
    pair_2_ajis = get_fast_aji(true, pred_2)
    pair_1_pq = get_fast_pq(true, pred_1)[0][-1]
    pair_2_pq = get_fast_pq(true, pred_2)[0][-1]
    print('True vs Pred 1: DICE 2: %0.4f AJI: %0.4f AJI+: %0.4f PQ: %0.4f' % (pair_1_dice_2, pair_1_ajis, pair_1_ajip, pair_1_pq))        
    print('True vs Pred 2: DICE 2: %0.4f AJI: %0.4f AJI+: %0.4f PQ: %0.4f' % (pair_2_dice_2, pair_2_ajis, pair_2_ajip, pair_2_pq))   

# ##
if __name__ == '__main__':
    compute_stat()
