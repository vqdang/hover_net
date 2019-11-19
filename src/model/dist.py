
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

"""
Ported from Naylor code at to match our processing frameworks
https://github.com/PeterJackNaylor/DRFNS/blob/master/src_RealData/
"""

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
        def down_conv_block(name, l, channel, nr_blks, stride=1):
            with tf.variable_scope(name):
                if stride != 1:
                    assert stride == 2, 'U-Net supports stride 2 down-sample only'
                    l = MaxPooling('max_pool', l, 2, strides=2)
                for idx in range(0, nr_blks):
                    l = Conv2D('conv_%d' % idx, l, channel, 3, 
                                padding='valid', strides=1, activation=BNReLU)
            return l
        ####
        def up_conv_block(name, l, shorcut, channel, nr_blks, stride=2):
            with tf.variable_scope(name):
                if stride != 1:
                    up_channel = l.get_shape().as_list()[1] # NCHW
                    assert stride == 2, 'U-Net supports stride 2 up-sample only'
                    l = Conv2DTranspose('deconv', l, up_channel, 2, strides=2)
                    l = tf.concat([l, shorcut], axis=1)
                for idx in range(0, nr_blks):
                    l = Conv2D('conv_%d' % idx, l, channel, 3, 
                                padding='valid', strides=1, activation=BNReLU)
            return l            
        ####
        is_training = get_current_tower_context().is_training
        
        images, truemap_coded = inputs

        orig_imgs = images

        if self.type_classification:
            true_type = truemap_coded[...,1]
            true_type = tf.cast(true_type, tf.int32)
            true_type = tf.identity(true_type, name='truemap-type')
            one_type  = tf.one_hot(true_type, self.nr_types, axis=-1)
            true_type = tf.expand_dims(true_type, axis=-1)

        true_dst = truemap_coded[...,-1]
        true_dst = tf.expand_dims(true_dst, axis=-1)
        true_dst = tf.identity(true_dst, name='truemap-dst')

        #### Xavier initializer
        with argscope(Conv2D, activation=tf.identity, use_bias=True,
                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                      bias_initializer=tf.constant_initializer(0.1)), \
                argscope([Conv2D, Conv2DTranspose, MaxPooling, BatchNorm], data_format=self.data_format):

            i = tf.transpose(images / 255.0, [0, 3, 1, 2])

            ####
            with tf.variable_scope('encoder'):
                e0 = down_conv_block('e0',  i,   32, nr_blks=2, stride=1) 
                e1 = down_conv_block('e1', e0,   64, nr_blks=2, stride=2) 
                e2 = down_conv_block('e2', e1,  128, nr_blks=2, stride=2) 
                e3 = down_conv_block('e3', e2,  256, nr_blks=2, stride=2) 
                e4 = down_conv_block('e4', e3,  512, nr_blks=2, stride=2) 

                c0 = crop_op(e0, (176, 176))
                c1 = crop_op(e1, (80, 80))
                c2 = crop_op(e2, (32, 32))
                c3 = crop_op(e3, (8, 8))

            with tf.variable_scope('decoder'):
                d3 = up_conv_block('d3', e4, c3, 256, nr_blks=2, stride=2)
                d2 = up_conv_block('d2', d3, c2, 128, nr_blks=2, stride=2)
                d1 = up_conv_block('d1', d2, c1,  64, nr_blks=2, stride=2)
                d0 = up_conv_block('d0', d1, c0,  32, nr_blks=2, stride=2)

            ####
            logi_dst = Conv2D('conv_out_dst', d0, 1, 1, activation=tf.identity)
            logi_dst = tf.transpose(logi_dst, [0, 2, 3, 1])
            pred_dst = tf.identity(logi_dst, name='predmap-dst')

            if self.type_classification:
                logi_type = Conv2D('conv_out_type', d0, self.nr_types, 1, activation=tf.identity)
                logi_type = tf.transpose(logi_type, [0, 2, 3, 1])
                soft_type = tf.nn.softmax(logi_type, axis=-1)
                # encoded so that inference can extract all output at once
                predmap_coded = tf.concat([soft_type, pred_dst], axis=-1)
            else:
                predmap_coded = pred_dst

            # * channel ordering: type-map, segmentation map
            # encoded so that inference can extract all output at once
            predmap_coded = tf.identity(predmap_coded, name='predmap-coded')

        ####
        if is_training:
            ######## LOSS
            loss = 0
            ### regression loss
            loss_mse = pred_dst - true_dst
            loss_mse = loss_mse * loss_mse
            loss_mse = tf.reduce_mean(loss_mse, name='loss_mse')
            loss += loss_mse

            if self.type_classification:
                loss_type = categorical_crossentropy(soft_type, one_type)
                loss_type = tf.reduce_mean(loss_type, name='loss-xentropy-class')
                add_moving_summary(loss_type)
                loss += loss_type

            wd_loss = regularize_cost('.*/W', l2_regularizer(5.0e-6), name='l2_regularize_loss')
            loss += wd_loss

            self.cost = tf.identity(loss, name='cost')
            add_moving_summary(self.cost)
            ####

            add_param_summary(('.*/W', ['histogram']))   # monitor W

            #### logging visual sthg
            orig_imgs = tf.cast(orig_imgs  , tf.uint8)
            tf.summary.image('input', orig_imgs, max_outputs=1)

            orig_imgs = crop_op(orig_imgs, (184, 184), "NHWC")

            pred_dst = colorize(pred_dst[...,0], cmap='jet')
            true_dst = colorize(true_dst[...,0], cmap='jet')

            viz = tf.concat([orig_imgs, true_dst, pred_dst,], 2)
            tf.summary.image('output', viz, max_outputs=1)

        return
