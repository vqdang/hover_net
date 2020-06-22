import cv2
import numpy as np
import time
import skimage
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.filters import rank, threshold_otsu
from scipy import ndimage
from scipy.ndimage.morphology import (binary_dilation, binary_erosion, binary_closing)

from .hover import proc_np_hv
from misc.utils import bounding_box
from misc.viz_utils import visualize_instances

####
def process_instance(pred_map, nr_types=None, overlaid_img=None, 
                     type_colour=None, output_dtype='uint16'):
    """
    Post processing script for image tiles

    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        nr_types: number of types considered at output of nc branch
        overlaid_img: img to overlay the predicted instances upon, `None` means no
        type_colour (dict) : `None` to use random, else overlay instances of a type to colour in the dict
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction 
    """

    if nr_types is not None:
        pred_inst = pred_map[..., nr_types:]
        pred_type = pred_map[..., :nr_types]
        pred_type = np.argmax(pred_type, axis=-1)
        pred_type = np.squeeze(pred_type)
    else:
        pred_inst = pred_map

    pred_inst = np.squeeze(pred_inst)
    pred_inst = proc_np_hv(pred_inst)
    pred_inst = pred_inst.astype(output_dtype)

    if nr_types is not None:
        pred_type_out = np.zeros([pred_type.shape[0], pred_type.shape[1]])               
        #### * Get class of each instance id, stored at index id-1
        pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
        pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
        for idx, inst_id in enumerate(pred_id_list):
            inst_tmp = pred_inst == inst_id
            inst_type = pred_type[pred_inst == inst_id]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0: # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            pred_type_out += (inst_tmp * inst_type)
        pred_type_out = pred_type_out.astype(output_dtype)
    else:
        pred_type_out = None
    
    if overlaid_img is not None:
        overlaid_img = visualize_instances(overlaid_img, pred_inst, 
                                pred_type_out, type_colour=type_colour)

    return pred_inst, pred_type_out, overlaid_img


####
def process_instance_wsi(pred_map, nr_types, tile_coords, return_masks, remap_label=False, offset=0, output_dtype='uint16'):
    """
    Post processing script

    Args:
        pred_map: commbined output of nc, np and hv branches
        nr_types: number of types considered at output of nc branch
        tile_coords: coordinates of top left corner of tile
        return_masks: whether to save cropped segmentation masks
        remap_label: whether to map instance labels from 1 to N (N = number of nuclei)
        offset: 
        output_dtype: data type of output
    
    Returns:
        mask_list_out: list of cropped predicted segmentation masks
        type_list_out: list of class predictions for each nucleus
        cent_list_out: list of centroid coordinates for each predicted instance (saved as (y,x))
    """

    # init output lists
    mask_list_out = []
    type_list_out = []
    cent_list_out = []

    pred_inst = pred_map[..., nr_types:] # output of combined np and hv branches
    pred_type = pred_map[..., :nr_types] # output of nc branch

    pred_inst = np.squeeze(pred_inst)
    pred_type = np.argmax(pred_type, axis=-1) # pixel wise class mask
    pred_type = np.squeeze(pred_type)

    pred_inst, pred_cent = proc_np_hv(pred_inst, return_coords=True)

    offset_x = tile_coords[0]+offset
    offset_y = tile_coords[1]+offset

    cent_list_out = [(x[0]+offset_y, x[1]+offset_x) for x in pred_cent] # ensure 
    
    # get the shape of the input tile
    shape_pred = pred_inst.shape
    
    # remap label is very slow - only uncomment if necessary to map labels in order
    if remap_label:
        pred_inst = remap_label(pred_inst, by_size=True)

    #### * Get class of each instance id, stored at index id-1
    pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
    for idx, inst_id in enumerate(pred_id_list):
        # crop the instance and type masks -> decreases search space and consequently computation time
        crop_inst, crop_type = crop_array(pred_inst, pred_type, pred_cent[idx], shape_pred)
        crop_inst_type = crop_type[crop_inst == inst_id]

        # get the masks cropped at the bounding box
        if return_masks:
            crop_inst_tmp = crop_inst == inst_id
            [rmin, rmax, cmin, cmax] = bounding_box(crop_inst_tmp)
            mask_bbox = crop_inst_tmp[rmin:rmax, cmin:cmax]
            mask_list_out.append(mask_bbox)

        # get the majority class within a given nucleus
        type_list, type_pixels = np.unique(crop_inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0: # ! if majority class is background, pick the 2nd most dominant class (if exists)
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        type_list_out.append(inst_type)

    return mask_list_out, type_list_out, cent_list_out


####
def crop_array(pred_inst, pred_type, pred_cent, shape_tile, crop_shape=(70,70)):
    """
    Crop the instance and class array with a given nucleus at the centre.
    Done to decrease the search space and consequently processing time.

    Args:
        pred_inst:  predicted nuclear instances for a given tile
        pred_type:  predicted nuclear types (pixel based) for a given tile
        pred_cent:  predicted centroid for a given nucleus
        shape_tile: shape of tile 
        crop_shape: output crop shape (saved as (y,x))

    Returns:
        crop_pred_inst: cropped pred_inst of shape crop_shape
        crop_pred_type: cropped pred_type of shape crop_shape
    """
    pred_x = pred_cent[1] # x coordinate
    pred_y = pred_cent[0] # y coordinate

    if pred_x < (crop_shape[0]/2):
        x_crop = 0
    elif pred_x > (shape_tile[1] - (crop_shape[1]/2)):
        x_crop = shape_tile[1] - crop_shape[1]
    else:
        x_crop = (pred_cent[1] - (crop_shape[1]/2))
    
    if pred_y < (crop_shape[0]/2):
        y_crop = 0
    elif pred_y > (shape_tile[0] - (crop_shape[0]/2)):
        y_crop = shape_tile[0] - crop_shape[0]
    else:
        y_crop = (pred_cent[0] - (crop_shape[0]/2))
    
    x_crop = int(x_crop)
    y_crop = int(y_crop)
    
    # perform the crop
    crop_pred_inst = pred_inst[y_crop:y_crop+crop_shape[1], x_crop:x_crop+crop_shape[0]]
    crop_pred_type = pred_type[y_crop:y_crop+crop_shape[1], x_crop:x_crop+crop_shape[0]]

    return crop_pred_inst, crop_pred_type


####
def img_min_axis(img):
    """
    Get the minimum of the x and y axes for an input array

    Args:
        img: input array
    """
    try:
        return min(img.shape[:2])
    except AttributeError:
        return min(img.size)


####
def stain_entropy_otsu(img):
    """
    Binarise an input image by calculating the entropy on the 
    hameatoxylin and eosin channels and then using otsu threshold 

    Args:
        img: input array
    """

    img_copy = img.copy()
    hed = skimage.color.rgb2hed(img_copy)  # convert colour space
    hed = (hed * 255).astype(np.uint8)
    h = hed[:, :, 0]
    e = hed[:, :, 1]
    d = hed[:, :, 2]
    selem = disk(4)  # structuring element
    # calculate entropy for each colour channel
    h_entropy = rank.entropy(h, selem)
    e_entropy = rank.entropy(e, selem)
    d_entropy = rank.entropy(d, selem)
    entropy = np.sum([h_entropy, e_entropy], axis=0) - d_entropy
    # otsu threshold
    threshold_global_otsu = threshold_otsu(entropy)
    mask = entropy > threshold_global_otsu

    return mask


####
def morphology(mask, proc_scale):
    """
    Applies a series of morphological operations
    to refine the binarised tissue mask

    Args:
        mask: input binary mask to refine
        proc_scale: scale at which to process
    
    Return:
        processed binary image
    """

    mask_scale = img_min_axis(mask)
    # Join together large groups of small components ('salt')
    radius = int(8 * proc_scale)
    selem = disk(radius)
    mask = binary_dilation(mask, selem)

    # Remove thin structures
    radius = int(16 * proc_scale)
    selem = disk(radius)
    mask = binary_erosion(mask, selem)

    # Remove small disconnected objects
    mask = remove_small_holes(
        mask,
        area_threshold=int(40 * proc_scale)**2,
        connectivity=1,
    )

    # Close up small holes ('pepper')
    mask = binary_closing(mask, selem)

    mask = remove_small_objects(
        mask,
        min_size=int(120 * proc_scale)**2,
        connectivity=1,
    )

    radius = int(16 * proc_scale)
    selem = disk(radius)
    mask = binary_dilation(mask, selem)

    mask = remove_small_holes(
        mask,
        area_threshold=int(40 * proc_scale)**2,
        connectivity=1,
    )

    # Fill holes in mask
    mask = ndimage.binary_fill_holes(mask)

    return mask


####
def get_tissue_mask(img, proc_scale=0.5):
    """
    Obtains tissue mask for a given image

    Args:
        img: input WSI as a np array
        proc_scale: scale at which to process
    
    Returns:
        binarised tissue mask
    """
    img_copy = img.copy()
    if proc_scale != 1.0:
        img_resize = cv2.resize(img_copy, None, fx=proc_scale, fy=proc_scale)
    else:
        img_resize = img_copy

    mask = stain_entropy_otsu(img_resize)
    mask = morphology(mask, proc_scale)
    mask = mask.astype('uint8')

    if proc_scale != 1.0:
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask


####
def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger instances has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
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
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1    
    return new_pred
