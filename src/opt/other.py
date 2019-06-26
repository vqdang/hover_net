import numpy as np
import tensorflow as tf
from .misc import * 

dist = {
    'train_input_shape' : [268, 268],
    'train_mask_shape'  : [ 84,  84],
    'infer_input_shape' : [268, 268],
    'infer_mask_shape'  : [ 84,  84], 

    'training_phase'    : [
        {
            'nr_epochs': 80,
            'manual_parameters' : { # Names of these variable defined within the Graph()
                'learning_rate': (1.0e-3, exp_decay_lr_schedule(80, 1.0e-3, 0.96, 10)), 
            },
            'train_batch_size'  : 8,
            'infer_batch_size'  : 16,

            'model_flags' : {
            }
        }
    ],

    'optimizer'           : tf.train.AdamOptimizer,

    'inf_batch_size'    : 16,
    'inf_auto_metric'   : 'valid_mse',
    'inf_auto_comparator' : '<',
}

unet = {
    'train_input_shape' : [268, 268],
    'train_mask_shape'  : [ 84,  84],
    'infer_input_shape' : [268, 268],
    'infer_mask_shape'  : [ 84,  84], 

    'training_phase'    : [
        {
            'nr_epochs': 240,
            'manual_parameters' : {
                'learning_rate': (1.0e-4, [('120', 1.0e-5)]), 
            },
            'train_batch_size'  : 8,
            'infer_batch_size'  : 16,

            'model_flags' : {
            }
        }
    ],

    'optimizer'           : tf.train.AdamOptimizer,

    'inf_batch_size'    : 16,
    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
}

fcn8 = {
    'train_input_shape' : [256, 256],
    'train_mask_shape'  : [256, 256],
    'infer_input_shape' : [256, 256],
    'infer_mask_shape'  : [256, 256], 

    'training_phase'    : [
        {
            'nr_epochs': 240,
            'manual_parameters' : {
                'learning_rate': (1.0e-4, [('120', 1.0e-5)]), 
            },
            'train_batch_size'  : 8,
            'infer_batch_size'  : 16,

            'model_flags' : {
            }
        }
    ],

    'optimizer'         : tf.train.AdamOptimizer,

    'inf_batch_size'    : 16,
    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
}

segnet = {
    'train_input_shape' : [256, 256],
    'train_mask_shape'  : [256, 256],
    'infer_input_shape' : [256, 256],
    'infer_mask_shape'  : [256, 256], 

    'training_phase'    : [
        {
            'nr_epochs': 240,
            'manual_parameters' : {
                'learning_rate': (1.0e-4, [('120', 1.0e-5)]), 
            },
            'train_batch_size'  : 8,
            'infer_batch_size'  : 16,

            'model_flags' : {
            }
        }
    ],

    'optimizer'         : tf.train.AdamOptimizer,
    'inf_batch_size'    : 16,
    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
}

dcan = {
    'train_input_shape' : [256, 256],
    'train_mask_shape'  : [256, 256],
    'infer_input_shape' : [256, 256],
    'infer_mask_shape'  : [256, 256], 

    'training_phase'    : [
        {
            'nr_epochs': 240,
            'manual_parameters' : {
                'learning_rate': (1.0e-4, [('120', 1.0e-5)]), 
                'aux_loss_dw'  : (1.0, 
                                [('50', 1.0e-1), ('100', 1.0e-2),
                                ('150', 1.0e-3), ('200', 1.0e-4)]),
            },
            'train_batch_size'  : 4,
            'infer_batch_size'  : 8,

            'model_flags' : {
            }
        }
    ],

    'optimizer'         : tf.train.AdamOptimizer,

    'inf_batch_size'  : 8,
    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
}

micronet = {
    'train_input_shape' : [252, 252],
    'train_mask_shape'  : [252, 252],
    'infer_input_shape' : [252, 252],
    'infer_mask_shape'  : [252, 252], 

    'training_phase'    : [
        {
            'nr_epochs': 250,
            'manual_parameters' : {
                'learning_rate': (1.0e-4, [('125', 1.0e-5)]),
                'aux_loss_dw'  : (1.0, 
                        [(str(epoch), 1.0 / epoch) for epoch in range(2, 251)]
                    ),
            },
            'train_batch_size'  : 4,
            'infer_batch_size'  : 8,

            'model_flags' : {
            }
        }
    ],

    'optimizer'         : tf.train.AdamOptimizer,

    'inf_batch_size'  : 8,
    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
}