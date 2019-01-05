
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorpack import imgaug

from loader.augs import (BinarizeLabel, GaussianBlur, GenInstanceDistance,
                         GenInstanceXY, MedianBlur)


#### 
class Config(object):
    def __init__(self):
        ####
        train_diff_shape = 190
        self.train_input_shape = [0, 0]
        self.train_mask_shape = [80, 80] # 
        self.train_input_shape[0] = self.train_mask_shape[0] + train_diff_shape
        self.train_input_shape[1] = self.train_mask_shape[1] + train_diff_shape
        ####
        infer_diff_shape = 190
        self.infer_input_shape = [0, 0]
        self.infer_mask_shape = [80, 80] 
        self.infer_input_shape[0] = self.infer_mask_shape[0] + infer_diff_shape
        self.infer_input_shape[1] = self.infer_mask_shape[1] + infer_diff_shape

        #### Training data
        self.do_valid = True
        
        self.data_ext = '.npy'
        self.train_dir = ['../../../train/Kumar/paper/train/XXXX/']
        self.valid_dir = ['../../train/Kumar/paper/valid_same/XXXX/',
                          '../../train/Kumar/paper/valid_diff/XXXX/']
        # nr of processes for parallel processing input
        self.nr_procs_train = 8 
        self.nr_procs_valid = 4 

        #### Training parameters
        ###
        # xy: double branches nework, 
        #     1 branch nuclei pixel classification (segmentation)
        #     1 branch regressing XY coordinate w.r.t the (supposed) 
        #     nearest nuclei centroids, coordinate is normalized to 0-1 range
        #
        # dst+np: double branches nework, 
        #     1 branch nuclei pixel classification (segmentation)
        #     1 branch regressing nuclei instance distance map (chessboard in this case),
        #     the distance map is normalized to 0-1 range

        self.model_mode  = 'xy'
        self.input_norm  = True # normalize RGB to 0-1 range
        self.loss_term   = ['bce', 'dice', 'mse', 'msge']

        #### 
        self.init_lr    = 1.0e-4
        self.nr_epochs  = 60
        self.lr_sched   = [('30', 1.0e-5)]
        self.nr_classes = 2        
        self.optim = tf.train.AdamOptimizer
        ####

        self.train_batch_size = 8
        self.infer_batch_size = 16

        ####
        exp_id = 'v6.0.3.0'
        model_id = '%s_temp' % (self.model_mode)
        self.model_name = '%s/%s' % (exp_id, model_id)
        # loading chkpts in tensorflow, the path must not contain extra '/'
        self.log_path = '/media/vqdang/Data_2/dang/output/NUC-SEG/collab'
        self.save_dir = '%s/%s/' % (self.log_path, self.model_name )

        self.pretrained_preact_resnet50_path = '../../pretrained/ImageNet-ResNet50-Preact.npz'

        ####
        metric_dict = {'valid_dice' : '>',
                       'valid_mse'  : '<'}

        self.inf_manual_chkpts = False
        self.inf_model_path  = self.save_dir + 'model-30420.index'
        # self.inf_eval_metric = 'valid_dice' if self.model_mode != 'xy' else 'valid_mse'
        self.inf_eval_metric = 'valid_dice'
        self.inf_comparator = metric_dict[self.inf_eval_metric]

        self.inf_imgs_ext      = '.png'
        self.inf_imgs_dir      = '../../data/NUC_TNBC/Images/XXXX/'
        self.inf_norm_root_dir = '../../data/NUC_TNBC/Images/'
        self.inf_norm_codes    = ['XXXX']
        self.inf_output_dir    = 'output/%s/TNBC/%s' % (exp_id, 'tune')

        #### name of nodes to extract output from GPU then feed to CPU
        # or feed input from CPU to GPU 

        # for inference during evalutaion mode i.e run by inferer.py
        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']
        # for inference during training mode i.e run by trainer.py
        self.train_inf_output_tensor_names = ['predmap-coded', 'truemap-coded']

        return

    def get_train_augmentors(self, view=False):
        shape_augs = [
            imgaug.Affine(
                        shear=5, # in degree
                        scale=(0.8, 1.2),
                        rotate_max_deg=179,
                        translate_frac=(0.01, 0.01),
                        interp=cv2.INTER_NEAREST,
                        border=cv2.BORDER_CONSTANT),
            imgaug.Flip(vert=True),
            imgaug.Flip(horiz=True),
            imgaug.CenterCrop(self.train_input_shape),
        ]

        input_augs = [
            imgaug.RandomApplyAug(
                imgaug.RandomChooseAug(
                    [
                    GaussianBlur(),
                    MedianBlur(),
                    imgaug.GaussianNoise(),
                    ]
                ), 0.5),
            # standard color augmentation
            imgaug.RandomOrderAug(
                [imgaug.Hue((-8, 8), rgb=True), # 0.04 * 90 degree
                imgaug.Saturation(0.2, rgb=True),
                imgaug.Brightness(26, clip=True), # 0.1 * 255 
                imgaug.Contrast((0.75, 1.25), clip=True),
                ]),
            imgaug.ToUint8(),
        ]

        # default to 'xy'
        if self.model_mode != 'dst+np':
            label_augs = [GenInstanceXY(self.train_mask_shape)]
        else:
            label_augs = [GenInstanceDistance(self.train_mask_shape)]
        label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(self.train_mask_shape))        

        return shape_augs, input_augs, label_augs

    def get_valid_augmentors(self, view=False):
        shape_augs = [
            imgaug.CenterCrop(self.train_input_shape),
        ]

        input_augs = None

        # default to 'xy'
        if self.model_mode != 'dst+np':
            label_augs = [GenInstanceXY(self.train_mask_shape)]
        else:
            label_augs = [GenInstanceDistance(self.train_mask_shape)]
        label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(self.train_mask_shape))        

        return shape_augs, input_augs, label_augs
