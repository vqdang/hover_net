import math
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet

from .net_utils import DenseBlock
from .utils import crop_op, crop_to_shape

sys.path.append("../../tiatoolbox")
from typing import List

from tiatoolbox.models.abc import ModelABC
from tiatoolbox.models.architecture.hovernet import HoVerNet
from tiatoolbox.models.architecture.utils import (UpSample2x, centre_crop,
                                                  centre_crop_to_shape)
from tiatoolbox.utils import misc
from tiatoolbox.utils.misc import get_bounding_box


class ResNetExt(ResNet):
    def _forward_impl(self, x, freeze):
        # See note [TorchScript super()]
        if self.training:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            with torch.set_grad_enabled(not freeze):
                x1 = x = self.layer1(x)
                x2 = x = self.layer2(x)
                x3 = x = self.layer3(x)
                x4 = x = self.layer4(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
        return x1, x2, x3, x4

    def forward(self, x: torch.Tensor, freeze: bool = False) -> torch.Tensor:
        return self._forward_impl(x, freeze)

    @staticmethod
    def resnet50(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [3, 4, 6, 3])
        model.conv1 = nn.Conv2d(
            num_input_channels, 64, 7, stride=1, padding=3)
        if pretrained is not None:
            pretrained = torch.load(pretrained)
            (
                missing_keys, unexpected_keys
            ) = model.load_state_dict(pretrained, strict=False)
        return model


####
class HoVerNetExt(ModelABC):
    """Initialise HoVer-Net."""

    def __init__(
            self,
            num_types=None,
            freeze=False,
            pretrained_backbone=None,
            ):
        super().__init__()
        self.freeze = freeze
        self.num_types = num_types
        self.output_ch = 3 if num_types is None else 4

        self.backbone = ResNetExt.resnet50(
            3, pretrained=pretrained_backbone)
        self.conv_bot = nn.Conv2d(
            2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            pad = ksize // 2
            module_list = [
                nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(256, [1, ksize], [128, 32], 8, split=4),
                nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            ]
            u3 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(128, [1, ksize], [128, 32], 4, split=4),
                nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            ]
            u2 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False),
            ]
            u1 = nn.Sequential(*module_list)

            module_list = [
                nn.BatchNorm2d(64, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
            ]
            u0 = nn.Sequential(*module_list)

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
            )
            return decoder

        ksize = 3
        if num_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=num_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()

    def forward(self, imgs):
        imgs = imgs / 255.0  # to 0-1 range to match XY

        d0, d1, d2, d3 = self.backbone(imgs, self.freeze)
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict

    @staticmethod
    def _create_decoder_branch(out_ch=2, ksize=5):
        """Helper to create a decoder branch."""
        modules = [
            ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
            ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
            (
                "convf",
                nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            ),
        ]
        u3 = nn.Sequential(OrderedDict(modules))

        modules = [
            ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
            ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
            (
                "convf",
                nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            ),
        ]
        u2 = nn.Sequential(OrderedDict(modules))

        modules = [
            ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
            (
                "conva",
                nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),
            ),
        ]
        u1 = nn.Sequential(OrderedDict(modules))

        modules = [
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
            (
                "conv",
                nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
            ),
        ]
        u0 = nn.Sequential(OrderedDict(modules))

        decoder = nn.Sequential(
            OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
        )
        return decoder

    @staticmethod
    def _proc_np_hv(np_map: np.ndarray, hv_map: np.ndarray, fx: float = 1):
        """Extract Nuclei Instance with NP and HV Map.

        Sobel will be applied on horizontal and vertical channel in
        `hv_map` to derive a energy landscape which highligh possible
        nuclei instance boundaries. Afterward, watershed with markers
        is applied on the above energy map using the `np_map` as filter
        to remove background regions.

        Args:
            np_map (np.ndarray): An image of shape (heigh, width, 1) which
              contains the probabilities of a pixel being a nuclei.
            hv_map (np.ndarray): An array of shape (heigh, width, 2) which
              contains the horizontal (channel 0) and vertical (channel 1)
              of possible instances exist withint the images.
            fx (float): The scale factor for processing nuclei. The scale
              assumes an image of resolution 0.25 microns per pixel. Default
              is therefore 1 for HoVer-Net.

        Returns:
            An np.ndarray of shape (height, width) where each non-zero values
            within the array correspond to one detected nuclei instances.

        """
        blb_raw = np_map[..., 0]
        h_dir_raw = hv_map[..., 0]
        v_dir_raw = hv_map[..., 1]

        # processing
        blb = np.array(blb_raw >= 0.5, dtype=np.int32)

        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1  # background is 0 already

        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        ksize = int((20 * fx) + 1)
        obj_size = math.ceil(10 * (fx ** 2))
        # Get resolution specific filters etc.

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        # * nuclei values form mountains so inverse to get basins
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=obj_size)

        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred

    @staticmethod
    def _get_instance_info(pred_inst, pred_type=None):
        """To collect instance information and store it within a dictionary.

        Args:
            pred_inst (np.ndarray): An image of shape (heigh, width) which
                contains the probabilities of a pixel being a nuclei.
            pred_type (np.ndarray): An image of shape (heigh, width, 1) which
                contains the probabilities of a pixel being a certain type of nuclei.

        Returns:
            inst_info_dict (dict): A dictionary containing a mapping of each instance
                    within `pred_inst` instance information. It has following form

                    inst_info = {
                            box: number[],
                            centroids: number[],
                            contour: number[][],
                            type: number,
                            prob: number,
                    }
                    inst_info_dict = {[inst_uid: number] : inst_info}

                    and `inst_uid` is an integer corresponds to the instance
                    having the same pixel value within `pred_inst`.

        """
        inst_id_list = np.unique(pred_inst)[1:]  # exclude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            inst_box = get_bounding_box(inst_map)
            inst_box_tl = inst_box[:2]
            inst_map = inst_map[inst_box[1] : inst_box[3], inst_box[0] : inst_box[2]]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = inst_contour[0][0].astype(np.int32)
            inst_contour = np.squeeze(inst_contour)

            # < 3 points does not make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small
            if inst_contour.shape[0] < 3:  # pragma: no cover
                continue
            # ! check for trickery shape
            if len(inst_contour.shape) != 2:  # pragma: no cover
                continue

            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour += inst_box_tl[None]
            inst_centroid += inst_box_tl  # X
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "box": inst_box,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "prob": None,
                "type": None,
            }

        if pred_type is not None:
            # * Get class of each instance id, stored at index id-1
            for inst_id in list(inst_info_dict.keys()):
                cmin, rmin, cmax, rmax = inst_info_dict[inst_id]["box"]
                inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
                inst_type_crop = pred_type[rmin:rmax, cmin:cmax]

                inst_map_crop = inst_map_crop == inst_id
                inst_type = inst_type_crop[inst_map_crop]

                (type_list, type_pixels) = np.unique(inst_type, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)

                inst_type = type_list[0][0]

                # ! pick the 2nd most dominant if it exists
                if inst_type == 0 and len(type_list) > 1:  # pragma: no cover
                    inst_type = type_list[1][0]

                type_dict = {v[0]: v[1] for v in type_list}
                type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)

                inst_info_dict[inst_id]["type"] = int(inst_type)
                inst_info_dict[inst_id]["prob"] = float(type_prob)

        return inst_info_dict

    @staticmethod
    # skipcq: PYL-W0221
    def postproc(raw_maps: List[np.ndarray]):
        """Post processing script for image tiles.

        Args:
            raw_maps (list(ndarray)): list of prediction output of each head and
                assumed to be in the order of [np, hv, tp] (match with the output
                of `infer_batch`).

        Returns:
            inst_map (ndarray): pixel-wise nuclear instance segmentation
                prediction.
            inst_dict (dict): a dictionary containing a mapping of each instance
                within `inst_map` instance information. It has following form

                inst_info = {
                    box: number[],
                    centroids: number[],
                    contour: number[][],
                    type: number,
                    prob: number,
                }
                inst_dict = {[inst_uid: number] : inst_info}

                and `inst_uid` is an integer corresponds to the instance
                having the same pixel value within `inst_map`.

        Examples:
            >>> from tiatoolbox.models.architecture.hovernet import HoVerNet
            >>> import torch
            >>> import numpy as np
            >>> batch = torch.from_numpy(image_patch)[None]
            >>> # image_patch is a 256x256x3 numpy array
            >>> weights_path = "A/weights.pth"
            >>> pretrained = torch.load(weights_path)
            >>> model = HoVerNet(num_types=6, mode="fast")
            >>> model.load_state_dict(pretrained)
            >>> output = model.infer_batch(model, batch, on_gpu=False)
            >>> output = [v[0] for v in output]
            >>> output = model.postproc(output)

        """
        if len(raw_maps) == 3:
            np_map, hv_map, tp_map = raw_maps
        else:
            tp_map = None
            np_map, hv_map = raw_maps

        pred_type = tp_map
        pred_inst = HoVerNet._proc_np_hv(np_map, hv_map)
        nuc_inst_info_dict = HoVerNet._get_instance_info(pred_inst, pred_type)

        return pred_inst, nuc_inst_info_dict

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch i/o
        aggregation.

        Args:
            model (nn.Module): PyTorch defined model.
            batch_data (ndarray): a batch of data generated by
                torch.utils.data.DataLoader.
            on_gpu (bool): Whether to run inference on a GPU.

        Returns:
            List of output from each head, each head is expected to contain
            N predictions for N input patches. There are two cases, one
            with 2 heads (Nuclei Pixels `np` and Hover `hv`) or with 2 heads
            (`np`, `hv`, and Nuclei Types `tp`).

        """
        patch_imgs = batch_data

        device = misc.select_device(on_gpu)
        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)  # to NCHW
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        model.eval()  # infer mode

        # --------------------------------------------------------------
        with torch.inference_mode():
            pred_dict = model(patch_imgs_gpu)
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
            )
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
            if "tp" in pred_dict:
                type_map = F.softmax(pred_dict["tp"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                type_map = type_map.type(torch.float32)
                pred_dict["tp"] = type_map
            pred_dict = {k: v.cpu().numpy() for k, v in pred_dict.items()}

        if "tp" in pred_dict:
            return pred_dict["np"], pred_dict["hv"], pred_dict["tp"]
        return pred_dict["np"], pred_dict["hv"]


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert f"Unknown Model Mode {mode}"
    return HoVerNet(mode=mode, **kwargs)
