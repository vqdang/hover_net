import os
import re
import glob
import json
import numpy as np
import shutil

from misc.utils import rm_n_mkdir, mkdir

####
def to_qupath(file_path, nuc_pos_list, nuc_type_list, type_info_dict):
    """
    For QuPath v0.2.3
    """
    nuc_pos_list = np.array(nuc_pos_list)
    nuc_type_list = np.array(nuc_type_list)
    assert nuc_pos_list.shape[0] == nuc_type_list.shape[0]
    with open(file_path, 'w') as fptr:
        fptr.write("x\ty\tclass\tname\tcolor\n")

        nr_nuc = nuc_pos_list.shape[0]
        for idx in range(nr_nuc):
            nuc_type = nuc_type_list[idx]
            nuc_pos = nuc_pos_list[idx]
            type_name = type_info_dict[nuc_type][0]
            type_color = type_info_dict[nuc_type][-1] # color in qupath format
            fptr.write('{x}\t{y}\t{type_class}\t{type_name}\t{type_color}\n'.format(
                        x=nuc_pos[0], y=nuc_pos[1], 
                        type_class='', 
                        type_name=type_name, 
                        type_color=type_color))
    return

####
if __name__ == '__main__':
    target_format = 'qupath'
    scale_factor = 0.5

    root_dir = 'dataset/rmt/continual_v0.0/pred/clf=[densenet-mini_v0.2.2]/'
    
    patterning = lambda x : re.sub('([\[\]])', '[\\1]', x)
    code_name_list = glob.glob(patterning('%s/*.json' % root_dir))
    code_name_list = [os.path.basename(v).split('.')[0] for v in code_name_list]
    code_name_list.sort()

    output_dir = root_dir

    for code_name in code_name_list[:]:
        nuc_info_path = '%s/%s.json' % (root_dir, code_name)
        if not os.path.exists(nuc_info_path): continue
        print(code_name)

        with open(nuc_info_path, "r") as handle:
            # TODO: reconcile format between tile mode and wsi mode
            # info_dict = json.load(handle)['nuc']
            info_dict = json.load(handle)

        new_info_dict = {}
        for inst_id, inst_info in info_dict.items():
            new_inst_info = {}
            for info_name, info_value in inst_info.items():
                if isinstance(info_value, list):
                    info_value = np.array(info_value)
                    info_value = info_value * scale_factor
                    info_value = info_value.astype(np.int32)
                new_inst_info[info_name] = info_value
            new_info_dict[inst_id] = new_inst_info

        centroid_list = np.array([v['centroid'] for v in list(new_info_dict.values())])
        type_list = np.array([v['type'] for v in list(new_info_dict.values())])

        type_info_dict = {
            0 : ('nolabe', (0  ,   0,   0), -1), # no label
            1 : ('neopla', (255,   0,   0), -65536), # neoplastic
            2 : ('inflam', (0  , 255,   0), -16711936), # inflamm
            3 : ('connec', (0  ,   0, 255), -16776961), # connective
            4 : ('necros', (255, 255,   0), -1), # dead
            5 : ('no-neo', (255, 165,   0), -1), # non-neoplastic epithelial
        }
        save_path = '%s/%s.tsv' % (output_dir, code_name)
        to_qupath(save_path, centroid_list, type_list, type_info_dict)
            
        # break