
import glob
import os
import shutil

import cv2
import numpy as np


####
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


####
def get_bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


####
def cropping_center(x, crop_shape, batch=False):
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    return x


####
def rm_n_mkdir(dir_path):
    if (os.path.isdir(dir_path)):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

####
def mkdir(dir_path):
    if (not os.path.isdir(dir_path)):
        os.makedirs(dir_path)

####
def get_files(data_dir_list, data_ext):
    """
    Given a list of directories containing data with extention 'data_ext',
    generate a list of paths for all files within these directories
    """

    data_files = []
    for sub_dir in data_dir_list:
        files_list = glob.glob(sub_dir + '/*' + data_ext)
        files_list.sort()  # ensure same order
        data_files.extend(files_list)

    return data_files


####
def get_inst_centroid(inst_map):
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [(inst_moment["m10"] / inst_moment["m00"]),
                         (inst_moment["m01"] / inst_moment["m00"])]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


####
def center_pad_to_shape(img, size, cval=255):
    # rounding down, add 1
    pad_h = size[0] - img.shape[0]
    pad_w = size[1] - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, 'constant', constant_values=cval)
    return img

####
def color_deconvolution(rgb, stain_mat):
    log255 = np.log(255) # to base 10, not base e
    rgb_float = rgb.astype(np.float64)
    log_rgb = -((255.0 * np.log((rgb_float+1) / 255.0)) / log255)
    output = np.exp(-(log_rgb @ stain_mat - 255.0) * log255 / 255.0)
    output[output > 255] = 255
    output = np.floor(output + 0.5).astype('uint8')
    return output

####
def check_available_subject(listA, listB):
    subject_code_setA = set(listA)
    subject_code_setB = set(listB)
    subject_code_list = set.intersection(subject_code_setA, subject_code_setB)
    subject_code_list = list(subject_code_list)
    subject_code_list.sort()
    in_A_not_in_B = list(subject_code_setA - subject_code_setB)
    in_B_not_in_A = list(subject_code_setB - subject_code_setA)
    in_A_not_in_B.sort()
    in_B_not_in_A.sort()
    # in_A_not_in_B, in_B_not_in_A, intersection
    return in_A_not_in_B, in_B_not_in_A, subject_code_list