

import cv2
import tensorflow as tf
from tensorpack import imgaug

from loader.augs import (BinarizeLabel, GaussianBlur, GenInstanceDistance,
                         GenInstanceXY, MedianBlur)

# TODO: make a general func to seed RNG
#### 
class Config(object):
    def __init__(self):
        #### Input - Output Height Width
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

        # patches are stored as numpy arrays with 4 channels
        # first 3 channels: image
        # final channel: instance label
        self.do_valid = True # run validation during training
        
        self.data_ext = '.npy' 
        self.train_dir = ['../../../train/Kumar/paper/train/XXXX/'] # directory of training patches
        self.valid_dir = ['../../../train/Kumar/paper/valid_same/XXXX/', 
                          '../../../train/Kumar/paper/valid_diff/XXXX/'] # directory of validation patches
        # nr of processes for parallel processing input
        self.nr_procs_train = 8 
        self.nr_procs_valid = 4 

        #### Training parameters
        ###
        # np+xy : double branches nework, 
        #     1 branch nuclei pixel classification (segmentation)
        #     1 branch regressing XY coordinate w.r.t the (supposed) 
        #     nearest nuclei centroids, coordinate is normalized to 0-1 range
        #
        # np+dst: double branches nework
        #     1 branch nuclei pixel classification (segmentation)
        #     1 branch regressing nuclei instance distance map (chessboard in this case),
        #     the distance map is normalized to 0-1 range

        self.model_mode  = 'np+xy'
        self.input_norm  = True # normalize RGB to 0-1 range
        #### Loss terms- refer to paper for more info
        # bce: binary cross entropy loss on the NP map
        # dice: soft dice loss on the NP map
        # mse: mean squared error on the XY map
        # msge: mean squared error on the gradient of XY map
        self.loss_term   = ['bce', 'dice', 'mse', 'msge'] 
        # np+dst run with 'bce', 'mse' flags

        #### 
        self.init_lr    = 1.0e-4
        self.nr_epochs  = 20
        self.lr_sched   = [('10', 1.0e-5)] 
        self.nr_classes = 2        
        self.optim = tf.train.AdamOptimizer
        ####

        self.train_phase1_batch_size = 8 # unfreezing everything will 
        self.train_phase2_batch_size = 4 # consume more memory
        self.infer_batch_size = 16

        ####
        exp_id = 'v1.0.0.0'
        model_id = '%s' % (self.model_mode)
        self.model_name = '%s/%s' % (exp_id, model_id)
        # loading chkpts in tensorflow, the path must not contain extra '/'
        self.log_path = 'chkpts/' # log root path
        self.save_dir = '%s/%s' % (self.log_path, self.model_name) # log file destination

        self.pretrained_preact_resnet50_path = '../../../pretrained/ImageNet-ResNet50-Preact.npz'

        ####      
        self.inf_model_path  = self.save_dir + '/tune/model-23998.index'

        #### Info for running inference
        self.inf_imgs_ext = '.tif'
        # 'inf_norm_root_dir' has various subdir each which contains a set of 
        # images such as orignal ('XXXX') or stain-normed wrt to a given target images
        # (like '5784' means TCGA-21-5784-01Z-00-DX1). The inference only run
        # with subdir provided by 'inf_norm_codes'
        self.inf_norm_root_dir = '../../../data/NUC_Kumar/train-set/imgs_norm/'
        self.inf_norm_codes    = ['XXXX']           
        self.inf_output_dir    = 'output/%s/Kumar/%s' % (exp_id, model_id)


        #### name of nodes to extract output from GPU then feed to CPU
        # or feed input from CPU to GPU 

        # for inference during evalutaion mode i.e run by inferer.py
        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']
        # for inference during training mode i.e run by trainer.py
        self.train_inf_output_tensor_names = ['predmap-coded', 'truemap-coded']

    # refer to https://tensorpack.readthedocs.io/modules/dataflow.imgaug.html for 
    # information on how to modify the augmentation parameters
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
                [imgaug.Hue((-8, 8), rgb=True), 
                imgaug.Saturation(0.2, rgb=True),
                imgaug.Brightness(26, clip=True),  
                imgaug.Contrast((0.75, 1.25), clip=True),
                ]),
            imgaug.ToUint8(),
        ]

        # default to 'xy'
        if self.model_mode != 'np+dst':
            label_augs = [GenInstanceXY(self.train_mask_shape)]
        else:
            label_augs = [GenInstanceDistance(self.train_mask_shape)]
        label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(self.train_mask_shape))        

        return shape_augs, input_augs, label_augs

    def get_valid_augmentors(self, view=False):
        shape_augs = [
            imgaug.CenterCrop(self.infer_input_shape),
        ]

        input_augs = None

        # default to 'xy'
        if self.model_mode != 'np+dst':
            label_augs = [GenInstanceXY(self.infer_mask_shape)]
        else:
            label_augs = [GenInstanceDistance(self.infer_mask_shape)]
        label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(self.infer_mask_shape))        

        return shape_augs, input_augs, label_augs
