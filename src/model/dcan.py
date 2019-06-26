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
                InputDesc(tf.float32, [None] + self.train_mask_shape  + [2], 'truemap-coded')]

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

        def proc_truemap(true, type_name):
            true = tf.cast(true, tf.int32)
            true = tf.identity(true, name='truemap-%s' % type_name)
            one = tf.one_hot(tf.squeeze(true), 2, axis=-1)
            return true, one

        orig_imgs = images
        obj_true = truemap_coded[...,0]
        obj_true = tf.expand_dims(obj_true, axis=-1)
        obj_true, obj_one = proc_truemap(obj_true, 'obj')

        cnt_true = truemap_coded[...,1]
        cnt_true = tf.expand_dims(cnt_true, axis=-1)
        cnt_true, cnt_one = proc_truemap(cnt_true, 'cnt')

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
            p4 = MaxPooling('pool4', d4, 2, strides=2, padding='valid')

            d5 = Conv2D('conv5_1',  p4, 512, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            d5 = Conv2D('conv5_2',  d5, 512, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            d5 = Conv2D('conv5_3',  d5, 512, 3, padding='same', strides=1, use_bias=True, activation=tf.nn.relu)
            p5 = MaxPooling('pool5', d5, 2, strides=2, padding='valid')

            d6 = Conv2D('conv6_1',  p5, 1024, 7, padding='same', use_bias=True, strides=1, activation=tf.nn.relu)
            d6 = tf.layers.dropout(d6, rate=0.5, seed=5, training=is_training)
            d6 = Conv2D('conv6_2',  d6, 1024, 1, padding='same', use_bias=True, strides=1, activation=tf.nn.relu)

            feat_scale_dict = {8 : d4, 16: d5, 32 : d6}

            # DECODERS ----------------------------------------------------------------------------------
            def decoder(name):             
                with tf.variable_scope('%s_decoder' % name):
                    soft_list = []
                    fused_logit = 0               
                    for idx, scale_lv in enumerate(feat_scale_dict.keys()):
                        logit = resize_op(feat_scale_dict[scale_lv], scale_lv, scale_lv, interp='bilinear', data_format='channels_first')
                        logit = Conv2D('upsample_%d' % idx, logit, self.nr_classes, 1, use_bias=True)
                        logit = tf.transpose(logit, [0, 2, 3, 1])
                        fused_logit += logit
                        soft_list.append(tf.nn.softmax(logit, axis=-1))
                    soft = tf.nn.softmax(fused_logit, axis=-1)
                    soft_list = soft_list + [soft]

                    prob = tf.identity(soft[..., 1], name='predmap-prob-fuse-%s' % name)
                    prob = tf.expand_dims(prob, axis=-1)
                    pred = tf.argmax(soft, axis=-1, name='predmap-%s' % name)
                    pred = tf.expand_dims(tf.cast(pred, tf.float32), axis=-1)
                return soft_list, prob, pred

            soft_obj_list, obj_prob, obj_pred = decoder('obj')
            soft_cnt_list, cnt_prob, cnt_pred = decoder('cnt')

            # --------------------------------------------------------------------------------------
            # encoded so that inference can extract all output at once
            predmap_coded = tf.concat([obj_prob, cnt_prob], axis=-1, name='predmap-coded')

        ####
        if is_training:
            ######## LOSS
            # get the variable to received fed weight from external scheduler
            with tf.variable_scope("", reuse=True):
                aux_loss_dw = tf.get_variable('aux_loss_dw')

            def compute_scale_loss_bce(soft_list, one, overall_loss, type_name):
                # last item in list is fused, so no decay
                for idx in range(len(soft_list)):
                    loss_bce = categorical_crossentropy(soft_list[idx], one)                
                    # NOTE: encodde last item in list is fused, so no decay
                    if idx != len(soft_list) - 1:
                        loss_bce = tf.reduce_mean(loss_bce, name='loss-bce-%s-us%d' % (type_name, idx))
                        overall_loss += aux_loss_dw * loss_bce
                        add_moving_summary(loss_bce)
                    else:
                        loss_bce = tf.reduce_mean(loss_bce, name='loss-bce-%s-fused' % type_name)
                        overall_loss += loss_bce
                        add_moving_summary(loss_bce)
                return overall_loss

            overall_loss = compute_scale_loss_bce(soft_obj_list, obj_one, 0, 'obj')
            overall_loss = compute_scale_loss_bce(soft_cnt_list, cnt_one, overall_loss, 'cnt')

            ####

            wd_loss = regularize_cost('.*/W', l2_regularizer(1.0e-5), name='l2_wd_loss')
            add_moving_summary(wd_loss)

            overall_loss = tf.identity(overall_loss + wd_loss, name='overall_loss')
            add_moving_summary(overall_loss)
            self.cost = overall_loss

            add_param_summary(('.*/W', ['histogram']))   # monitor W

            #### logging visual sthg
            orig_imgs = tf.cast(orig_imgs  , tf.uint8)
            tf.summary.image('input', orig_imgs, max_outputs=1)
    
            obj_pred = colorize(obj_prob[...,0], cmap='jet')
            obj_true = colorize(obj_true[...,0], cmap='viridis')

            cnt_pred = colorize(cnt_prob[...,0], cmap='jet')
            cnt_true = colorize(cnt_true[...,0], cmap='viridis')

            viz = tf.concat([orig_imgs, 
                        obj_pred, obj_true,
                        cnt_pred, cnt_true], 2)

            tf.summary.image('output', viz, max_outputs=1)

        return
