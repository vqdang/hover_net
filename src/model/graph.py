
import tensorflow as tf

from tensorpack import *
from tensorpack.models import BatchNorm, BNReLU, Conv2D, MaxPooling, FixedUnPooling
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary


from .utils import *

import sys
sys.path.append("..") # adds higher directory to python modules path.
try: # HACK: import beyond current level, may need to restructure
    from config import Config
except ImportError:
    assert False, 'Fail to import config.py'

class Model(ModelDesc, Config):
    def __init__(self, freeze=False):
        super(Model, self).__init__()
        assert tf.test.is_gpu_available()
        self.freeze = freeze
        self.data_format = 'NCHW'

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None] + self.train_input_shape + [3], 'images'),
                InputDesc(tf.float32, [None] + self.train_mask_shape  + [3], 'truemap-coded')]
          
    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.init_lr, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = self.optim(learning_rate=lr)
        return opt

    def _build_graph(self, inputs):
        ####
        def upsample2x(name, x):
            """
            Nearest neighbor up-sampling
            """
            return FixedUnPooling(
                        name, x, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
                        data_format='channels_first')
        ####
        def res_blk(name, l, ch, ksize, count, split=1, strides=1):
            ch_in = l.get_shape().as_list()
            with tf.variable_scope(name):
                for i in range(0, count):
                    with tf.variable_scope('block' + str(i)):  
                        x = l if i == 0 else BNReLU('preact', l)
                        x = Conv2D('conv1', x, ch[0], ksize[0], activation=BNReLU)
                        x = Conv2D('conv2', x, ch[1], ksize[1], split=split, 
                                        strides=strides if i == 0 else 1, activation=BNReLU)
                        x = Conv2D('conv3', x, ch[2], ksize[2], activation=tf.identity)
                        if (strides != 1 or ch_in[1] != ch[2]) and i == 0:
                            l = Conv2D('convshortcut', l, ch[2], 1, strides=strides)
                        x = tf.stop_gradient(x) if self.freeze else x
                        l = l + x
                # end of each group need an extra activation
                l = BNReLU('bnlast',l)  
            return l
        ####
        def dense_blk(name, l, ch, ksize, count, split=1, padding='valid'):
            with tf.variable_scope(name):
                for i in range(0, count):
                    with tf.variable_scope('blk/' + str(i)):
                        x = BNReLU('preact_bna', l)
                        x = Conv2D('conv1', x, ch[0], ksize[0], padding=padding, activation=BNReLU)
                        x = Conv2D('conv2', x, ch[1], ksize[1], padding=padding, split=split)
                        ##
                        if padding == 'valid':
                            x_shape = x.get_shape().as_list()
                            l_shape = l.get_shape().as_list()
                            l = crop_op(l, (l_shape[2] - x_shape[2], 
                                            l_shape[3] - x_shape[3]))

                        l = tf.concat([l, x], axis=1)
                l = BNReLU('blk_bna', l)
            return l
        ####
        
        images, truemap_coded = inputs

        orig_imgs = images

        o_true_blb = truemap_coded[...,0]
        o_true_blb = tf.cast(o_true_blb, tf.int32)
        o_true_blb = tf.identity(o_true_blb, name='truemap-blb')
        o_one_blb  = tf.one_hot(o_true_blb, 2, axis=-1)
        o_true_blb = tf.expand_dims(o_true_blb, axis=-1)

        o_true_xy = truemap_coded[...,1:]
        o_true_xy = tf.identity(o_true_xy, name='truemap-xy')

        ####
        def encoder(i):
            d1 = Conv2D('conv0',  i, 64, 7, padding='valid', strides=1, activation=BNReLU)
            d1 = res_blk('group0', d1, [ 64,  64,  256], [1, 3, 1], 3, strides=1)                       
            
            d2 = res_blk('group1', d1, [128, 128,  512], [1, 3, 1], 4, strides=2)
            d2 = tf.stop_gradient(d2) if self.freeze else d2

            d3 = res_blk('group2', d2, [256, 256, 1024], [1, 3, 1], 6, strides=2)
            d3 = tf.stop_gradient(d3) if self.freeze else d3

            d4 = res_blk('group3', d3, [512, 512, 2048], [1, 3, 1], 3, strides=2)
            d4 = tf.stop_gradient(d4) if self.freeze else d4
            
            d4 = Conv2D('conv_bot',  d4, 1024, 1, padding='same')
            return [d1, d2, d3, d4]
        ####
        def decoder(name, i):
            ####
            pad = 'valid'
            with tf.variable_scope(name):
                with tf.variable_scope('u3'):
                    u3 = upsample2x('rz', i[-1])
                    u3_sum = tf.add_n([u3, i[-2]])

                    u3 = Conv2D('conva', u3_sum, 256, 5, strides=1, padding=pad)   
                    u3 = dense_blk('dense', u3, [128, 32], [1, 5], 8, split=4, padding=pad)
                    u3 = Conv2D('convf', u3, 512, 1, strides=1)   
                ####
                with tf.variable_scope('u2'):          
                    u2 = upsample2x('rz', u3)
                    u2_sum = tf.add_n([u2, i[-3]])

                    u2 = Conv2D('conva', u2_sum, 128, 5, strides=1, padding=pad)
                    u2 = dense_blk('dense', u2, [128, 32], [1, 5], 4, split=4, padding=pad)
                    u2 = Conv2D('convf', u2, 256, 1, strides=1)   
                ####
                with tf.variable_scope('u1'):          
                    u1 = upsample2x('rz', u2)
                    u1_sum = tf.add_n([u1, i[-4]])

                    u1 = u1_sum # for legacy

            return u1
        ####
        with argscope(Conv2D, activation=tf.identity, use_bias=False, # K.he initializer
                      W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')), \
                argscope([Conv2D, BatchNorm], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0

            ####
            d = encoder(i)
            d[0] = crop_op(d[0], (184, 184))
            d[1] = crop_op(d[1], (72, 72))

            ####
            o_blb = decoder('blb', d)
            o_blb = BNReLU('preact_out_blb', o_blb)

            o_xy  = decoder('xy', d)
            o_xy  = BNReLU('preact_out_xy', o_xy)

            ####
            o_logi_blb = Conv2D('conv_out_blb', o_blb, 2, 1, use_bias=True, activation=tf.identity)
            o_logi_blb = tf.transpose(o_logi_blb, [0, 2, 3, 1])
            o_soft_blb = tf.nn.softmax(o_logi_blb, axis=-1)
            o_prob_blb = tf.identity(o_soft_blb[...,1], name='predmap-prob-blb')
            o_prob_blb = tf.expand_dims(o_prob_blb, axis=-1)
            o_pred_blb = tf.argmax(o_soft_blb, axis=-1, name='predmap-blb')
            o_pred_blb = tf.expand_dims(tf.cast(o_pred_blb, tf.float32), axis=-1)

            o_logi_xy = Conv2D('conv_out_xy', o_xy, 2, 1, use_bias=True, activation=tf.identity)
            o_logi_xy = tf.transpose(o_logi_xy, [0, 2, 3, 1])
            o_prob_xy = tf.identity(o_logi_xy, name='predmap-prob-xy')
            o_pred_xy = tf.identity(o_logi_xy, name='predmap-xy')

            # encoded so that inference can extract all output at once
            predmap_coded = tf.concat([o_prob_blb, o_pred_xy], axis=-1, name='predmap-coded')
        ####

        ####
        if get_current_tower_context().is_training:
            ######## LOSS
            ### XY regression loss
            loss_mse = o_pred_xy - o_true_xy
            loss_mse = loss_mse * loss_mse
            loss_mse = tf.reduce_mean(loss_mse, name='loss_mse')
            add_moving_summary(loss_mse)

            loss_xy = loss_mse
            if 'msge' in self.loss_term:
                nuclear = truemap_coded[...,0]
                nuclear = tf.stack([nuclear, nuclear], axis=-1)
                pred_grad = get_gradient_xy(o_pred_xy, 1, 0)
                true_grad = get_gradient_xy(o_true_xy, 1, 0) 
                loss_msge = pred_grad - true_grad
                loss_msge = nuclear * (loss_msge * loss_msge)
                # artificial reduce_mean with focus region
                loss_msge = tf.reduce_sum(loss_msge)
                loss_msge = loss_msge / tf.reduce_sum(nuclear) 
                loss_msge = tf.identity(loss_msge, name='loss_msge')
                add_moving_summary(loss_msge)

                loss_xy = 2 * loss_mse + loss_msge
                
            loss_xy = tf.identity(loss_xy, name='overall-xy')

            ### Nuclei Blob classification loss
            loss_bce = categorical_crossentropy(o_soft_blb, o_one_blb)
            loss_bce = tf.reduce_mean(loss_bce, name='loss-bce')
            add_moving_summary(loss_bce)

            loss_blb = loss_bce
            if 'dice' in self.loss_term:
                loss_dice = dice_loss(o_prob_blb, o_true_blb)
                loss_dice = tf.identity(loss_dice, name='loss-dice')
                add_moving_summary(loss_dice)

                loss_blb = loss_bce + loss_dice

            loss_blb = tf.identity(loss_blb, name='overall-blb')

            ### combine the loss into single cost function
            self.cost = tf.identity(loss_xy + loss_blb, name='overall-loss')            
            add_moving_summary(self.cost, loss_xy, loss_blb)
            ####

            add_param_summary(('.*/W', ['histogram']))   # monitor W

            #### logging visual sthg
            orig_imgs = tf.cast(orig_imgs  , tf.uint8)
            tf.summary.image('input', orig_imgs, max_outputs=1)

            orig_imgs = crop_op(orig_imgs, (190, 190), "NHWC")

            o_pred = colorize(o_prob_blb[...,0], cmap='jet')
            o_true = colorize(o_true_blb[...,0], cmap='jet')
            o_pred = tf.cast(o_pred * 255, tf.uint8)
            o_true = tf.cast(o_true * 255, tf.uint8)

            pred_x = colorize(o_prob_xy[...,0], vmin=-1, vmax=1, cmap='jet')
            pred_y = colorize(o_prob_xy[...,1], vmin=-1, vmax=1, cmap='jet')
            true_x = colorize(o_true_xy[...,0], vmin=-1, vmax=1, cmap='jet')
            true_y = colorize(o_true_xy[...,1], vmin=-1, vmax=1, cmap='jet')
            pred_x = tf.cast(pred_x * 255, tf.uint8)
            pred_y = tf.cast(pred_y * 255, tf.uint8)
            true_x = tf.cast(true_x * 255, tf.uint8)
            true_y = tf.cast(true_y * 255, tf.uint8)

            viz = tf.concat([orig_imgs, 
                            pred_x, pred_y, o_pred, 
                            true_x, true_y, o_true], 2)

            tf.summary.image('output', viz, max_outputs=1)

        return