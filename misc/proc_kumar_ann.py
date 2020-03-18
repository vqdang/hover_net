
##
## For processing .xml annotation file in Kumar dataset
##

import glob
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .utils import rm_n_mkdir

###
imgs_dir = "" # .tif 
anns_dir = "" # .xml folders
save_dir = "" # storing .npy ground truth (HW)

file_list = glob.glob(imgs_dir + '*.tif')
file_list.sort() # ensure same order [1]

rm_n_mkdir(save_dir)

for filename in file_list: # png for base
    filename = os.path.basename(filename)
    basename = filename.split('.')[0]

    print(basename)
    img = cv2.imread(imgs_dir + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hw = img.shape[:2]

    xml = ET.parse(anns_dir + basename + '.xml')

    contour_dbg = np.zeros(hw, np.uint8)

    insts_list = []
    for idx, region_xml in enumerate(xml.findall('.//Region')):
        vertices = []
        for vertex_xml in region_xml.findall('.//Vertex'):
            attrib = vertex_xml.attrib
            vertices.append([float(attrib['X']), 
                             float(attrib['Y'])])
        vertices = np.array(vertices) + 0.5
        vertices = vertices.astype('int32')
        contour_blb = np.zeros(hw, np.uint8)
        # fill both the inner area and contour with idx+1 color
        cv2.drawContours(contour_blb, [vertices], 0, idx+1, -1)
        insts_list.append(contour_blb)

    insts_size_list = np.array(insts_list)
    insts_size_list = np.sum(insts_size_list, axis=(1 , 2))
    insts_size_list = list(insts_size_list)

    pair_insts_list = zip(insts_list, insts_size_list)
    # sort in z-axis basing on size, larger on top
    pair_insts_list = sorted(pair_insts_list, key=lambda x: x[1])
    insts_list, insts_size_list = zip(*pair_insts_list)

    ann = np.zeros(hw, np.int32)
    for idx, inst_map in enumerate(insts_list):
        ann[inst_map > 0] = idx + 1

    np.save('%s/%s.npy' % (save_dir, basename), ann)
