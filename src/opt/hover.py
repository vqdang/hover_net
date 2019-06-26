
import tensorflow as tf
from .misc import * 

#### Training parameters
###
# np+hv : double branches nework, 
#     1 branch nuclei pixel classification (segmentation)
#     1 branch regressing horizontal/vertical coordinate w.r.t the (supposed) 
#     nearest nuclei centroids, coordinate is normalized to 0-1 range
#
# np+dst: double branches nework
#     1 branch nuclei pixel classification (segmentation)
#     1 branch regressing nuclei instance distance map (chessboard in this case),
#     the distance map is normalized to 0-1 range

np_hv = {
    'train_input_shape' : [270, 270],
    'train_mask_shape'  : [ 80,  80],
    'infer_input_shape' : [270, 270],
    'infer_mask_shape'  : [ 80,  80], 

    'training_phase'    : [
        {
            'nr_epochs': 50,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('25', 1.0e-5)]), 
            },
            'pretrained_path'  : '../../../pretrained/ImageNet-ResNet50-Preact.npz',
            'train_batch_size' : 8,
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : True
            }
        },

        {
            'nr_epochs': 50,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('25', 1.0e-5)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : -1,
            'train_batch_size' : 4, # unfreezing everything will
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : False
            }
        }
    ],

    'loss_term' : {'bce' : 1, 'dice' : 1, 'mse' : 2, 'msge' : 1}, 

    'optimizer'           : tf.train.AdamOptimizer,

    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
    'inf_batch_size' : 16,
}

np_dist = {
    'train_input_shape' : [270, 270],
    'train_mask_shape'  : [ 80,  80],
    'infer_input_shape' : [270, 270],
    'infer_mask_shape'  : [ 80,  80], 

    'training_phase'    : [
        {
            'nr_epochs': 50,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('25', 1.0e-5)]), 
            },
            'pretrained_path'  : '../../../pretrained/ImageNet-ResNet50-Preact.npz',
            'train_batch_size' : 8,
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : True
            }
        },

        {
            'nr_epochs': 50,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('25', 1.0e-5)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : -1,
            'train_batch_size' : 4, # unfreezing everything will
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : False
            }
        }
    ],
  
    'optimizer'         : tf.train.AdamOptimizer,

    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
    'inf_batch_size' : 16,
}
