import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from .utils import crop_op, crop_to_shape
from config import Config


####
class Net(nn.Module):
    """ A base class provides a common weight initialisation scheme."""

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x


####
class TFSamepaddingLayer(nn.Module):
    """To align with tf `same` padding. 
    
    Putting this before any conv layer that need padding
    Assuming kernel has Height == Width for simplicity
    """

    def __init__(self, ksize, stride):
        super(TFSamepaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
        # print(x.shape, padding)
        x = F.pad(x, padding, "constant", 0)
        # print(x.shape)
        return x


####
class DenseBlock(Net):
    """Dense Block as defined in:

    Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. 
    "Densely connected convolutional networks." In Proceedings of the IEEE conference 
    on computer vision and pattern recognition, pp. 4700-4708. 2017.

    Only performs `valid` convolution.

    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super(DenseBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("preact_bna/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                            ("preact_bna/relu", nn.ReLU(inplace=True)),
                            (
                                "conv1",
                                nn.Conv2d(
                                    unit_in_ch,
                                    unit_ch[0],
                                    unit_ksize[0],
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),
                            ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                            ("conv1/relu", nn.ReLU(inplace=True)),
                            # ('conv2/pool', TFSamepaddingLayer(ksize=unit_ksize[1], stride=1)),
                            (
                                "conv2",
                                nn.Conv2d(
                                    unit_ch[0],
                                    unit_ch[1],
                                    unit_ksize[1],
                                    groups=split,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),
                        ]
                    )
                )
            )
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1]

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = crop_to_shape(prev_feat, new_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat


####
class ResidualBlock(Net):
    """Residual block as defined in:

    He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning 
    for image recognition." In Proceedings of the IEEE conference on computer vision 
    and pattern recognition, pp. 770-778. 2016.

    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, stride=1):
        super(ResidualBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            unit_layer = [
                ("preact/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                ("preact/relu", nn.ReLU(inplace=True)),
                (
                    "conv1",
                    nn.Conv2d(
                        unit_in_ch,
                        unit_ch[0],
                        unit_ksize[0],
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                ("conv1/relu", nn.ReLU(inplace=True)),
                (
                    "conv2/pad",
                    TFSamepaddingLayer(
                        ksize=unit_ksize[1], stride=stride if idx == 0 else 1
                    ),
                ),
                (
                    "conv2",
                    nn.Conv2d(
                        unit_ch[0],
                        unit_ch[1],
                        unit_ksize[1],
                        stride=stride if idx == 0 else 1,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("conv2/bn", nn.BatchNorm2d(unit_ch[1], eps=1e-5)),
                ("conv2/relu", nn.ReLU(inplace=True)),
                (
                    "conv3",
                    nn.Conv2d(
                        unit_ch[1],
                        unit_ch[2],
                        unit_ksize[2],
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                ),
            ]
            # * has bna to conclude each previous block so
            # * must not put preact for the first unit of this block
            unit_layer = unit_layer if idx != 0 else unit_layer[2:]
            self.units.append(nn.Sequential(OrderedDict(unit_layer)))
            unit_in_ch = unit_ch[-1]

        if in_ch != unit_ch[-1] or stride != 1:
            self.shortcut = nn.Conv2d(in_ch, unit_ch[-1], 1, stride=stride, bias=False)
        else:
            self.shortcut = None

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        # print(self.units[0])
        # print(self.units[1])
        # exit()

    def out_ch(self):
        return self.unit_ch[-1]

    def forward(self, prev_feat, freeze=False):
        if self.shortcut is None:
            shortcut = prev_feat
        else:
            shortcut = self.shortcut(prev_feat)

        for idx in range(0, len(self.units)):
            new_feat = prev_feat
            if self.training:
                with torch.set_grad_enabled(not freeze):
                    new_feat = self.units[idx](new_feat)
            else:
                new_feat = self.units[idx](new_feat)
            prev_feat = new_feat + shortcut
            shortcut = prev_feat
        feat = self.blk_bna(prev_feat)
        return feat


####
class UpSample2x(nn.Module):
    """Upsample input by a factor of 2.
    
    Assume input is of NCHW, port FixedUnpooling from TensorPack.
    """

    def __init__(self):
        super(UpSample2x, self).__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        # unsqueeze is expand_dims equivalent
        # permute is transpose equivalent
        # view is reshape equivalent
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret

