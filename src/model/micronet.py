
import tensorflow as tf

from tensorpack import *
from tensorpack.models import BatchNorm, BNReLU, Conv2D, MaxPooling, Conv2DTranspose
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary

from .utils import *

import sys
sys.path.append("..") # adds higher directory to python modules path.
try: # HACK: import beyond current level, may need to restructure
    from config import Config
except ImportError:
    assert False, 'Fail to import config.py'


class Graph(ModelDesc, Config):
    def __init__(self):
        super(Graph, self).__init__()
        assert tf.test.is_gpu_available()
        self.data_format = 'channels_first'

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None] + self.train_input_shape + [3], 'images'),
                InputDesc(tf.float32, [None] + self.train_mask_shape  + [None], 'truemap-coded')]

    # for node to receive manual info such as learning rate.
    def add_manual_variable(self, name, init_value, summary=True):
        var = tf.get_variable(name, initializer=init_value, trainable=False)
        if summary:
            tf.summary.scalar(name + '-summary', var)
        return

    def _get_optimizer(self):
        with tf.variable_scope("", reuse=True):
            lr = tf.get_variable('learning_rate')
        opt = self.optimizer(learning_rate=lr)
        return opt

    def _build_graph(self, inputs):

        ####
        is_training = get_current_tower_context().is_training
        
        images, truemap_coded = inputs

        orig_imgs = images
        pen_map = truemap_coded[...,-1]
        if self.type_classification:
            true = truemap_coded[...,1]
        else:
            true = truemap_coded[...,0]            
        true = tf.cast(true, tf.int32)
        true = tf.identity(true, name='truemap')
        one  = tf.one_hot(true, self.nr_types if self.type_classification else self.nr_classes, axis=-1)
        true = tf.expand_dims(true, axis=-1)

        def down_branch(name, main_in, aux_in, ch):
            with tf.variable_scope(name):
                a = Conv2D('conv1', main_in, ch, 3, padding='valid', use_bias=False, activation=BNReLU)
                a = Conv2D('conv2', a, ch, 3, padding='valid', use_bias=True, activation=tf.nn.relu)
                a = MaxPooling('pool', a, 2, strides=2, padding= 'same') 

                b = Conv2D('conv3', aux_in, ch, 3, padding='valid', use_bias=False, activation=BNReLU)
                b = Conv2D('conv4',      b, ch, 3, padding='valid', use_bias=True, activation=tf.nn.relu)

                c = tf.concat([a, b], axis=1)
            return c

        def up_branch(name, main_in, aux_in, ch):
            with tf.variable_scope(name):
                a = Conv2DTranspose('up1', main_in, ch, 2, strides=(2, 2), padding='same', use_bias=True, activation=tf.identity)
                a = Conv2D('conv1', a, ch, 3, padding='valid', use_bias=True, activation=tf.nn.relu)
                a = Conv2D('conv2', a, ch, 3, padding='valid', use_bias=True, activation=tf.nn.relu)

                # stride 1 is no different from normal 5x5 conv, 'valid' to gain extrapolated border pixels
                b1 = Conv2DTranspose('up2',      a, ch, 5, strides=(1, 1), padding='valid', use_bias=True, activation=tf.identity)
                b2 = Conv2DTranspose('up3', aux_in, ch, 5, strides=(1, 1), padding='valid', use_bias=True, activation=tf.identity)
                b = tf.concat([b1, b2], axis=1)
                b = Conv2D('conv3', b, ch, 1, padding='same', use_bias=True, activation=tf.nn.relu)
            return b

        def aux_branch(name, main_in, up_kernel, up_strides):
            ch = main_in.get_shape().as_list()[1] # NCHW
            with tf.variable_scope(name): # preserve the depth
                a = Conv2DTranspose('up', main_in, ch, up_kernel, strides=up_strides, padding='same', use_bias=True, activation=tf.identity)
                a = Conv2D('conv', a, self.nr_types if self.type_classification else self.nr_classes, 3, padding='valid', activation=tf.nn.relu)
                a = tf.layers.dropout(a, rate=0.5, seed=5, training=is_training)
            return a

        #### Xavier initializer
        with argscope(Conv2D, activation=tf.identity, 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True),
                    bias_initializer=tf.constant_initializer(0.1)), \
             argscope(Conv2DTranspose, activation=tf.identity, 
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True),
                    bias_initializer=tf.constant_initializer(0.1)), \
                argscope([Conv2D, Conv2DTranspose, MaxPooling, BatchNorm], data_format=self.data_format):

            i = tf.transpose(images / 255.0, [0, 3, 1, 2]) # our way
            resize_func = lambda x, y: resize_op(x, size=y,interp='bicubic', data_format='channels_first')

            ####
            b1 = down_branch('b1',  i, resize_func(i, (128, 128)),  64)
            b2 = down_branch('b2', b1, resize_func(i, ( 64,  64)), 128)
            b3 = down_branch('b3', b2, resize_func(i, ( 32,  32)), 256)
            b4 = down_branch('b4', b3, resize_func(i, ( 16,  16)), 512)

            with tf.variable_scope('b5'):
                b5 = Conv2D('conv1', b4, 2048, 3, padding='valid', use_bias=True, activation=tf.nn.relu)
                b5 = Conv2D('conv2', b5, 2048, 3, padding='valid', use_bias=True, activation=tf.nn.relu)
            b6 = up_branch('b6', b5, b4, 1024)
            b7 = up_branch('b7', b6, b3, 512)
            b8 = up_branch('b8', b7, b2, 256)
            b9 = up_branch('b9', b8, b1, 128)

            aux_out1 = aux_branch('aux_out1', b9, 2, (2, 2))
            aux_out2 = aux_branch('aux_out2', b8, 4, (4, 4))
            aux_out3 = aux_branch('aux_out3', b7, 8, (8, 8))
            out = tf.concat([aux_out1, aux_out2, aux_out3], axis=1)
            out_list = [out, aux_out1, aux_out2, aux_out3]

            soft_list = []
            prob_list = []
            for idx, sub_out in enumerate(out_list):
                logi = Conv2D('conv_out%d' % idx, sub_out, 
                                self.nr_types if self.type_classification else self.nr_classes, 
                                3, padding='valid', use_bias=True, activation=tf.identity)
                logi = tf.transpose(logi, [0, 2, 3, 1])
                soft = tf.nn.softmax(logi, axis=-1)

                if self.type_classification:
                    prob_np = tf.reduce_sum(soft[...,1:], axis=-1, keepdims=True)
                    prob_np = tf.identity(prob_np, name='predmap-prob-np')
                else:
                    prob_np = tf.identity(soft[...,1], name='predmap-prob')
                    prob_np = tf.expand_dims(prob_np, axis=-1)
                
                soft_list.append(soft)
                prob_list.append(prob_np)

            # return the aggregated output
            # encoded so that inference can extract all output at once
            if self.type_classification:
                predmap_coded = tf.concat([soft_list[0], prob_list[0]], axis=-1, name='predmap-coded')
            else:
                predmap_coded = tf.identity(prob_list[0], name='predmap-coded')

        ####
        if is_training:
            ######## LOSS                       
            # get the variable to received fed weight from external scheduler
            with tf.variable_scope("", reuse=True):
                aux_loss_dw = tf.get_variable('aux_loss_dw')

            loss_list = [] # index 0 is main output
            global_step = tf.train.get_or_create_global_step()
            global_step = tf.cast(global_step, tf.float32)
            for idx, sub_soft in enumerate(soft_list):
                loss_bce = categorical_crossentropy(sub_soft, one)
                loss_bce = tf.reduce_mean(loss_bce * pen_map)
                loss_bce = loss_bce if idx == 0 else loss_bce * aux_loss_dw
                loss_bce = tf.identity(loss_bce, name='loss-bce-%d' % idx)
                loss_list.append(loss_bce)
                add_moving_summary(loss_bce)

            wd_loss = regularize_cost('.*/W', l2_regularizer(1.0e-5), name='l2_wd_loss')
            add_moving_summary(wd_loss)

            cost = tf.add_n(loss_list) + wd_loss
            self.cost = tf.identity(cost, name='overall_cost')
            add_moving_summary(self.cost)
            ####

            add_param_summary(('.*/W', ['histogram']))   # monitor W

            #### logging visual sthg
            orig_imgs = tf.cast(orig_imgs  , tf.uint8)
            tf.summary.image('input', orig_imgs, max_outputs=1)

            colored_list = [true] + prob_list + [tf.expand_dims(pen_map, axis=-1)]
            colored_list = [colorize(feat[...,0], cmap='jet') for feat in colored_list]

            viz = tf.concat([orig_imgs] + colored_list, 2)
            tf.summary.image('output', viz, max_outputs=1)

        return
