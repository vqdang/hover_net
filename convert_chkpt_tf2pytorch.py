"""convert_chkpt_tf2pytorch.py.

Convert checkpoint from previous repository trained with tensorflow to pytorch.

"""

import pandas as pd
import numpy as np
import torch

model_mode = "Pytorch-Fast"  # or 'Pytorch-Fast' vs 'Pytorch-Original'
mapping = pd.read_csv("tf_to_pytorch_variable_mapping.csv", sep="\t", index_col=False)
# mapping = {v["Tensorflow"]: v[model_mode] for k, v in mapping.T.to_dict().items()}
# mapping

# tf_path = "../pretrained/ImageNet-ResNet50-Preact.npz"
# pt_path = "../pretrained/ImageNet-ResNet50-Preact_pytorch.tar"

# tf_path = '../pretrained/hover_seg_Kumar.npz'
# pt_path = "../pretrained/hovernet_original_kumar_pytorch.tar"

# tf_path = '../pretrained/hover_seg_CoNSeP.npz'
# pt_path = "../pretrained/hovernet_original_consep_pytorch.tar"

# tf_path = '../pretrained/hover_seg_CPM17.npz'
# pt_path = "../pretrained/hovernet_original_cpm17_pytorch.tar"

# tf_path = "../pretrained/hover_seg_&_class_CoNSeP.npz"
# pt_path = "../pretrained/hovernet_original_consep_type-pytorch.tar"

# tf_path = "../pretrained/pecan-hover-net.npz"
# pt_path = "../pretrained/hovernet_fast_pannuke_pytorch.tar"

# tf_path = "../pretrained/hovernet_fast_monusac_tf.npz"
# pt_path = "../pretrained/hovernet_fast_monusac_pytorch.tar"

# pt_path = 'dumped_pytorch_chkpt.tar'

# pt = {}
# tf = np.load(tf_path)

# for tf_k, tf_v in tf.items():
#     if 'linear' in tf_k: continue # should only be for pretrained model 
#     pt_k = mapping[tf_k]
#     if "conv" in pt_k and "bn" not in pt_k and "bias" not in pt_k:
#         tf_v = np.transpose(tf_v, [3, 2, 0, 1])
#     if "shortcut" in pt_k:
#         tf_v = np.transpose(tf_v, [3, 2, 0, 1])
#     pt[pt_k] = torch.from_numpy(tf_v)
# # make compatible with repo structure
# pt["upsample2x.unpool_mat"] = torch.from_numpy(np.ones((2, 2), dtype="float32"))
# pt = {"desc": pt}
# torch.save(pt, pt_path)

# pt_path = "../pretrained/consep_net_epoch=40.npz"

tf_path = "../pretrained/consep_net_epoch=40.tar"
pt_path = "../pretrained/hovernet_original_consep_native.tar"

# tf_path = "../pretrained/kumar_net_epoch=44.tar"
# pt_path = "../pretrained/hovernet_original_kumar_native.tar"

mapping = {v["Pytorch-Original"]: v['Pytorch-Fast'] for k, v in mapping.T.to_dict().items()}

pt = {}
ptx = torch.load(tf_path)['desc']
for tf_k, tf_v in ptx.items():
    if 'num_batches_tracked' in tf_k: continue
    if tf_k == 'upsample2x.unpool_mat': 
        pt[tf_k] = tf_v
        continue
    pt_k = mapping[tf_k]
    pt[pt_k] = tf_v
# make compatible with repo structure
pt = {"desc": pt}
torch.save(pt, pt_path)
