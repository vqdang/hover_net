"""convert_chkpt_tf2pytorch.py.

Convert checkpoint from previous repository trained with tensorflow to pytorch.

"""

import pandas as pd
import numpy as np
import torch

model_mode = "Pytorch-Original"  # or 'Pytorch-Fast'
mapping = pd.read_csv("tf_to_pytorch_variable_mapping.csv", sep="\t", index_col=False)
mapping = {v["Tensorflow"]: v[model_mode] for k, v in mapping.T.to_dict().items()}
# mapping

# tf_path = '../pretrained/pecan-hover-net.npz'
tf_path = "../pretrained/hover_seg_&_class_CoNSeP.npz"
# tf_path = '../pretrained/hover_seg_CoNSeP.npz'
pt_path = "../pretrained/hover_seg_consep_type-pytorch.tar"

# pt_path = 'dumped_pytorch_chkpt.tar'

pt = {}
tf = np.load(tf_path)

for tf_k, tf_v in tf.items():
    pt_k = mapping[tf_k]
    if "conv" in pt_k and "bn" not in pt_k and "bias" not in pt_k:
        tf_v = np.transpose(tf_v, [3, 2, 0, 1])
    if "shortcut" in pt_k:
        tf_v = np.transpose(tf_v, [3, 2, 0, 1])
    pt[pt_k] = torch.from_numpy(tf_v)
# make compatible with repo structure
pt["upsample2x.unpool_mat"] = torch.from_numpy(np.ones((2, 2), dtype="float32"))
pt = {"desc": pt}
torch.save(pt, pt_path)

