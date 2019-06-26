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

        ####
        def unpool_with_argmax(pool, ind, name=None, ksize=[1, 2, 2, 1]):
            """
            Unpooling layer after max_pool_with_argmax.
            Args:
                pool: max pooled output tensor, must be NCHW
                ind: argmax indices
                ksize:ksize is the same as for the pool
            Return:
                unpool: unpooling tensor
            """
            pool = tf.transpose(pool, [0, 2, 3, 1])
            with tf.variable_scope(name):
                input_shape = tf.cast(tf.shape(pool), tf.int64)
                output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]
                output_shape = tf.stack(output_shape)

                flat_input_size = tf.reduce_prod(input_shape)
                flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

                pool_ = tf.reshape(pool, [-1]) # simply flatten
                batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[-1, 1, 1, 1])
                b = tf.ones_like(ind) * batch_range
                b = tf.expand_dims(tf.reshape(b, [-1]), axis=-1)
                ind_ = tf.expand_dims(tf.reshape(ind, [-1]), axis=-1)
                ind_ = tf.concat([b, ind_], 1)

                ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
                ret = tf.reshape(ret, shape=output_shape)

                input_shape =  pool.get_shape().as_list()
                ret.set_shape((None,
                    input_shape[1] * 2 if input_shape[1] is not None else None,
                    input_shape[2] * 2 if input_shape[2] is not None else None,
                    input_shape[3] if input_shape[3] is not None else None))

            ret = tf.transpose(ret, [0, 3, 1, 2])
            return ret
        ####
        def maxpool_with_argmax(x, kernel_size, stride, name=None, padding='SAME'):
            """
            Max-pooling layer. Downsamples values within a kernel to the 
            maximum value within the corresponding kernel
            Args:
                x: Input to the max-pooling layer, must be NCHW
                kernel_size: Size of kernel where max-pooling is applied
                stride: Determines the downsample factor
                name: Name scope for operation
                padding: Same or valid padding
                index: Boolean- whether to return pooling indicies
            Return:
                pool: Tensor with max-pooling
                argmax: Indicies of maximal values computed in each kernel (use with segnet)
            """
            strides = [1,stride,stride,1]
            ksize = [1, kernel_size, kernel_size, 1]
            x = tf.transpose(x, [0, 2, 3, 1])
            pool, argmax = tf.nn.max_pool_with_argmax(x, ksize, strides,
                            padding=padding, name=name)
            pool = tf.transpose(pool, [0, 3, 1, 2])
            return pool, argmax

        #### Xavier initializer
        with argscope(Conv2D, activation=tf.identity, use_bias=False,
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()), \
                argscope([Conv2D, MaxPooling, AvgPooling, BatchNorm], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0

            # ENCODER -----------------------------------------------------------------------------------------

            # NOTE: dont need bias when applying BN after Conv
            d1 = Conv2D('conv1_1',  i, 64, 3, padding='same', strides=1, activation=BNReLU)
            d1 = Conv2D('conv1_2', d1, 64, 3, padding='same', strides=1, activation=BNReLU)
            p1, p1_idx = maxpool_with_argmax(d1, 2, 2, name='pool1', padding='SAME')

            d2 = Conv2D('conv2_1',  p1, 128, 3, padding='same', strides=1, activation=BNReLU)
            d2 = Conv2D('conv2_2',  d2, 128, 3, padding='same', strides=1, activation=BNReLU)
            p2, p2_idx = maxpool_with_argmax(d2, 2, 2, name='pool2', padding='SAME')

            d3 = Conv2D('conv3_1',  p2, 256, 3, padding='same', strides=1, activation=BNReLU)
            d3 = Conv2D('conv3_2',  d3, 256, 3, padding='same', strides=1, activation=BNReLU)
            d3 = Conv2D('conv3_3',  d3, 256, 3, padding='same', strides=1, activation=BNReLU)
            p3, p3_idx = maxpool_with_argmax(d3, 2, 2, name='pool3', padding='SAME')

            d4 = Conv2D('conv4_1',  p3, 512, 3, padding='same', strides=1, activation=BNReLU)
            d4 = Conv2D('conv4_2',  d4, 512, 3, padding='same', strides=1, activation=BNReLU)
            d4 = Conv2D('conv4_3',  d4, 512, 3, padding='same', strides=1, activation=BNReLU)
            p4, p4_idx = maxpool_with_argmax(d4, 2, 2, name='pool4', padding='SAME')

            d5 = Conv2D('conv5_1',  p4, 512, 3, padding='same', strides=1, activation=BNReLU)
            d5 = Conv2D('conv5_2',  d5, 512, 3, padding='same', strides=1, activation=BNReLU)
            d5 = Conv2D('conv5_3',  d5, 512, 3, padding='same', strides=1, activation=BNReLU)
            p5, p5_idx = maxpool_with_argmax(d5, 2, 2, name='pool5', padding='SAME')

            # DECODER -----------------------------------------------------------------------------------------

            us1 = unpool_with_argmax(p5, ind=p5_idx, name='unpool_1')
            us1 = Conv2D('conv_us1_1',  us1, 512, 3, padding='same', strides=1, activation=BNReLU)
            us1 = Conv2D('conv_us1_2',  us1, 512, 3, padding='same', strides=1, activation=BNReLU)
            us1 = Conv2D('conv_us1_3',  us1, 512, 3, padding='same', strides=1, activation=BNReLU)

            us2 = unpool_with_argmax(us1, ind=p4_idx, name='unpool_2')
            us2 = Conv2D('conv_us2_1',  us2, 512, 3, padding='same', strides=1, activation=BNReLU)
            us2 = Conv2D('conv_us2_2',  us2, 512, 3, padding='same', strides=1, activation=BNReLU)
            us2 = Conv2D('conv_us2_3',  us2, 256, 3, padding='same', strides=1, activation=BNReLU)

            us3 = unpool_with_argmax(us2, ind=p3_idx, name='unpool_3')
            us3 = Conv2D('conv_us3_1',  us3, 256, 3, padding='same', strides=1, activation=BNReLU)
            us3 = Conv2D('conv_us3_2',  us3, 256, 3, padding='same', strides=1, activation=BNReLU)
            us3 = Conv2D('conv_us3_3',  us3, 128, 3, padding='same', strides=1, activation=BNReLU)

            us4 = unpool_with_argmax(us3, ind=p2_idx, name='unpool_4')
            us4 = Conv2D('conv_us4_1',  us4, 128, 3, padding='same', strides=1, activation=BNReLU)
            us4 = Conv2D('conv_us4_2',  us4,  64, 3, padding='same', strides=1, activation=BNReLU)

            us5 = unpool_with_argmax(us4, ind=p1_idx, name='unpool_5')
            us5 = Conv2D('conv_us5_1',  us5, 64, 3, padding='same', strides=1, activation=BNReLU)
            us5 = Conv2D('conv_us5_2',  us5, self.nr_classes, 1, padding='same', strides=1, use_bias=True, activation=tf.identity)

            logi = tf.transpose(us5, [0, 2, 3, 1])
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
