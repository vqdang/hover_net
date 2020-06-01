
import cv2
import math
import random
import colorsys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm

from .utils import bounding_box

from config import Config


####
def visualize_instances(input_image, pred_inst, pred_type=None, colours=None, line_thickness=2):
    """
    Overlays segmentation results on image as contours

    Args:
        input_image: input image
        pred_inst: instance mask with unique value for every object
        pred_type: type mask with unique value for every class
        colours: 
        line_thickness: line thickness of contours

    Returns:
        overlay: output image with segmentation overlay as contours
    """
    overlay = np.copy((input_image).astype(np.uint8))

    if pred_type is not None:
        type_list = list(np.unique(pred_type))  # get list of types
        type_list.remove(0)  # remove background
    else:
        type_list = [1]

    for type_id in type_list:
        if pred_type is not None:
            label_map = (pred_type == type_id) * pred_inst
        else:
            label_map = pred_inst
        inst_list = list(np.unique(label_map))  # get list of instances
        inst_list.remove(0)  # remove background
        contours = []
        for inst_id in inst_list:
            inst_map = np.array(
                pred_inst == inst_id, np.uint8)  # get single object
            y1, y2, x1, x2 = bounding_box(inst_map)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= pred_inst.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= pred_inst.shape[0] - 1 else y2
            inst_map_crop = inst_map[y1:y2, x1:x2]
            contours_crop = cv2.findContours(
                inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            index_correction = np.asarray([[[[x1, y1]]]])
            for i in range(len(contours_crop[0])):
                contours.append(
                    list(np.asarray(contours_crop[0][i].astype('int32')) + index_correction))
        contours = list(itertools.chain(*contours))
        cv2.drawContours(overlay, np.asarray(contours), -1,
                         colours[type_id], line_thickness)
    return overlay


####
def gen_figure(imgs_list, titles, fig_inch, shape=None,
               share_ax='all', show=False, colormap=plt.get_cmap('jet')):

    num_img = len(imgs_list)
    if shape is None:
        ncols = math.ceil(math.sqrt(num_img))
        nrows = math.ceil(num_img / ncols)
    else:
        nrows, ncols = shape

    # generate figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=share_ax, sharey=share_ax)
    axes = [axes] if nrows == 1 else axes

    # not very elegant
    idx = 0
    for ax in axes:
        for cell in ax:
            cell.set_title(titles[idx])
            cell.imshow(imgs_list[idx], cmap=colormap)
            cell.tick_params(axis='both',
                             which='both',
                             bottom='off',
                             top='off',
                             labelbottom='off',
                             right='off',
                             left='off',
                             labelleft='off')
            idx += 1
            if idx == len(titles):
                break
        if idx == len(titles):
            break

    fig.tight_layout()
    return fig
