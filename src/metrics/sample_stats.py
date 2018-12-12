
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

    dice_true   = cv2.imread('dice_true.png', cv2.IMREAD_GRAYSCALE)
    dice_pred_1 = cv2.imread('dice_pred_1.png', cv2.IMREAD_GRAYSCALE)
    dice_pred_2 = cv2.imread('dice_pred_2.png', cv2.IMREAD_GRAYSCALE)
    print('DICE2 Sample: True vs Pred 1 ', get_dice_2(dice_true, dice_pred_1))        
    print('DICE2 Sample: True vs Pred 2 ', get_dice_2(dice_true, dice_pred_2))        

    ##
    aji_true = cv2.imread('aji_true.png', cv2.IMREAD_GRAYSCALE)
    aji_pred = cv2.imread('aji_pred.png', cv2.IMREAD_GRAYSCALE)

    aji_true_1 = np.copy(aji_true)
    aji_true_2 = np.copy(aji_true)

    aji_true_1[aji_true_1 ==  76] = 1
    aji_true_1[aji_true_1 == 149] = 2
    aji_true_1[aji_true_1 == 225] = 3

    aji_true_2[aji_true_2 ==  76] = 2
    aji_true_2[aji_true_2 == 149] = 3
    aji_true_2[aji_true_2 == 225] = 1

    aji_pred[aji_pred ==  76] = 1
    aji_pred[aji_pred == 149] = 2

    print('AJI Sample: True 1 vs Pred ', get_aji(aji_true_1, aji_pred))        
    print('AJI Sample: True 2 vs Pred ', get_aji(aji_true_2, aji_pred))        
    
##
if __name__ == '__main__':
    compute_stat()
