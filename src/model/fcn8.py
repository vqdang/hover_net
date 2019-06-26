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
                InputDesc(tf.float32, [None] + self.train_mask_shape  + [1], 'truemap-coded')]

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

        is_training = get_current_tower_context().is_training

        images, truemap_coded = inputs

        orig_imgs = images
        true = truemap_coded[...,0]
        true = tf.cast(true, tf.int32)
        true = tf.identity(true, name='truemap')
        one  = tf.one_hot(true, 2, axis=-1)
        true = tf.expand_dims(true, axis=-1)

        #### Xavier initializer
        with argscope(Conv2D, activation=tf.identity, use_bias=False,
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()), \
                argscope([Conv2D, MaxPooling, AvgPooling, BatchNorm], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0

            d1 = Conv2D('conv1_1',  i, 64, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            d1 = Conv2D('conv1_2', d1, 64, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            p1 = MaxPooling('pool1', d1, 2, strides=2, padding= 'valid')  

            d2 = Conv2D('conv2_1',  p1, 128, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            d2 = Conv2D('conv2_2',  d2, 128, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            p2 = MaxPooling('pool2', d2, 2, strides=2, padding= 'valid')  

            d3 = Conv2D('conv3_1',  p2, 256, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            d3 = Conv2D('conv3_2',  d3, 256, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            d3 = Conv2D('conv3_3',  d3, 256, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            p3 = MaxPooling('pool3', d3, 2, strides=2, padding= 'valid')  

            d4 = Conv2D('conv4_1',  p3, 512, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            d4 = Conv2D('conv4_2',  d4, 512, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            d4 = Conv2D('conv4_3',  d4, 512, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            p4 = MaxPooling('pool4', d4, 2, strides=2, padding= 'valid')  

            d5 = Conv2D('conv5_1',  p4, 512, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            d5 = Conv2D('conv5_2',  d5, 512, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            d5 = Conv2D('conv5_3',  d5, 512, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            p5 = MaxPooling('pool5', d5, 2, strides=2, padding= 'valid')  

            d6 = Conv2D('conv6_1',  p5, 4096, 7, padding='same', use_bias=True, strides=1, activation=tf.nn.relu)
            d6 = tf.layers.dropout(d6, rate=0.5, seed=5, training=is_training)
            d6 = Conv2D('conv6_2',  d6, 4096, 1, padding='same', use_bias=True, strides=1, activation=tf.nn.relu)
            d6 = tf.layers.dropout(d6, rate=0.5, seed=5, training=is_training)
            d6 = Conv2D('conv6_3',  d6, self.nr_classes, 1, padding='same', use_bias=True, strides=1)

            score_pool4 = Conv2D('score_pool4', p4, self.nr_classes, 1, use_bias=True)
            upsample1 = resize_op(d6, 2, 2, interp='bilinear', data_format='channels_first')
            fuse1 = tf.add_n([upsample1, score_pool4])

            score_pool3 = Conv2D('score_pool3', p3, self.nr_classes, 1, use_bias=True)
            upsample2 = resize_op(fuse1, 2, 2, interp='bilinear', data_format='channels_first')
            fuse2 = tf.add_n([upsample2, score_pool3])
            fuse2 = resize_op(fuse2, 8, 8, interp='bilinear', data_format='channels_first')

            logi = tf.transpose(fuse2, [0, 2, 3, 1])
            soft = tf.nn.softmax(logi, axis=-1)

            prob = tf.identity(soft[...,1], name='predmap-prob')
            prob = tf.expand_dims(prob, axis=-1)
            pred = tf.argmax(soft, axis=-1, name='predmap')
            pred = tf.expand_dims(tf.cast(pred, tf.float32), axis=-1)

            # encoded so that inference can extract all output at once
            predmap_coded = tf.concat([prob], axis=-1, name='predmap-coded')

        ####
        if is_training:
            ######## LOSS
            ### classification loss
            loss_bce = categorical_crossentropy(soft, one)
            loss_bce = tf.reduce_mean(loss_bce, name='loss-bce')
            add_moving_summary(loss_bce)

            wd_loss = regularize_cost('.*/W', l2_regularizer(1.0e-5), name='l2_wd_loss')
            add_moving_summary(wd_loss)
            self.cost = loss_bce + wd_loss

            add_param_summary(('.*/W', ['histogram']))   # monitor W

            #### logging visual sthg
            orig_imgs = tf.cast(orig_imgs  , tf.uint8)
            tf.summary.image('input', orig_imgs, max_outputs=1)
    
            pred = colorize(prob[...,0], cmap='jet')
            true = colorize(true[...,0], cmap='jet')
            viz = tf.concat([orig_imgs, pred, true], 2)

            tf.summary.image('output', viz, max_outputs=1)

        return