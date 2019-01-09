
import cv2
import math
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

####
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

####
def visualize_instances(mask, canvas=None, color=None):
    """
    Args:
        mask: array of NW
    Return:
        Image with the instance overlaid
    """

    canvas = np.full(mask.shape + (3,), 200, dtype=np.uint8) \
                if canvas is None else np.copy(canvas)

    insts_list = list(np.unique(mask))
    insts_list.remove(0) # remove background

    inst_colors = random_colors(len(insts_list))
    inst_colors = np.array(inst_colors) * 255

    for idx, inst_id in enumerate(insts_list):
        inst_color = color if color is not None else inst_colors[idx]
        inst_map = np.array(mask == inst_id, np.uint8)
        contours = cv2.findContours(inst_map.copy(), 
                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours[1], -1, inst_color, 2)
    return canvas

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
####