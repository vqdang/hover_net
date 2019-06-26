
import math
import numpy as np

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from matplotlib import cm

# TODO: assert for data format
####
def resize_op(x, height_factor=None, width_factor=None, size=None, 
                interp='bicubic', data_format='channels_last'):
    """
    Resize by a factor if `size=None` else resize to `size`
    """
    original_shape = x.get_shape().as_list()
    if size is not None:
        if data_format == 'channels_first':
            x = tf.transpose(x, [0, 2, 3, 1])
            if interp == 'bicubic':
                x = tf.image.resize_bicubic(x, size)
            elif interp == 'bilinear':
                x = tf.image.resize_bilinear(x, size)
            else:
                x = tf.image.resize_nearest_neighbor(x, size)
            x = tf.transpose(x, [0, 3, 1, 2])
            x.set_shape((None, 
                original_shape[1] if original_shape[1] is not None else None, 
                size[0], size[1]))
        else:
            if interp == 'bicubic':
                x = tf.image.resize_bicubic(x, size)
            elif interp == 'bilinear':
                x = tf.image.resize_bilinear(x, size)
            else:
                x = tf.image.resize_nearest_neighbor(x, size)
            x.set_shape((None, 
                size[0], size[1], 
                original_shape[3] if original_shape[3] is not None else None))
    else:
        if data_format == 'channels_first':
            new_shape = tf.cast(tf.shape(x)[2:], tf.float32)    
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('float32'))
            new_shape = tf.cast(new_shape, tf.int32)    
            x = tf.transpose(x, [0, 2, 3, 1])
            if interp == 'bicubic':
                x = tf.image.resize_bicubic(x, new_shape)
            elif interp == 'bilinear':
                x = tf.image.resize_bilinear(x, new_shape)
            else:
                x = tf.image.resize_nearest_neighbor(x, new_shape)
            x = tf.transpose(x, [0, 3, 1, 2])
            x.set_shape((None,
                        original_shape[1] if original_shape[1] is not None else None,
                        int(original_shape[2] * height_factor) if original_shape[2] is not None else None,
                        int(original_shape[3] * width_factor) if original_shape[3] is not None else None))
        else:
            original_shape = x.get_shape().as_list()
            new_shape = tf.cast(tf.shape(x)[1:3], tf.float32)    
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('float32'))
            new_shape = tf.cast(new_shape, tf.int32)    
            if interp == 'bicubic':
                x = tf.image.resize_bicubic(x, new_shape)
            elif interp == 'bilinear':
                x = tf.image.resize_bilinear(x, new_shape)
            else:
                x = tf.image.resize_nearest_neighbor(x, new_shape)
            x.set_shape((None,
                        int(original_shape[1] * height_factor) if original_shape[1] is not None else None,
                        int(original_shape[2] * width_factor) if original_shape[2] is not None else None,
                        original_shape[3] if original_shape[3] is not None else None))
    return x 

####
def crop_op(x, cropping, data_format='channels_first'):
    """
    Center crop image
    Args:
        cropping is the substracted portion
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == 'channels_first':
        x = x[:,:,crop_t:-crop_b,crop_l:-crop_r]
    else:
        x = x[:,crop_t:-crop_b,crop_l:-crop_r]
    return x       
####

def categorical_crossentropy(output, target):
    """
        categorical cross-entropy, accept probabilities not logit
    """
    # scale preds so that the class probs of each sample sum to 1
    output /= tf.reduce_sum(output,
                            reduction_indices=len(output.get_shape()) - 1,
                            keepdims=True)
    # manual computation of crossentropy
    epsilon = tf.convert_to_tensor(10e-8, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    return - tf.reduce_sum(target * tf.log(output),
                            reduction_indices=len(output.get_shape()) - 1)
####
def dice_loss(output, target, loss_type='sorensen', axis=None, smooth=1e-3):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), 
              dice = ```smooth/(small_value + smooth)``, then if smooth is very small, 
              dice close to 0 (even the image values lower than the threshold), 
              so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> dice_loss = dice_coe(outputs, y_)
    """
    target = tf.squeeze(tf.cast(target, tf.float32))
    output = tf.squeeze(tf.cast(output, tf.float32))

    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknown loss_type")
    # already flatten
    dice = 1.0 - (2. * inse + smooth) / (l + r + smooth)
    ##
    return dice
####
def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    Arguments:
      - value: input tensor, NHWC ('channels_last')
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3], uint8.
    """

    # normalize
    if vmin is None:
        vmin = tf.reduce_min(value, axis=[1,2])
        vmin = tf.reshape(vmin, [-1, 1, 1])
    if vmax is None:
        vmax = tf.reduce_max(value, axis=[1,2])
        vmax = tf.reshape(vmax, [-1, 1, 1])
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    # NOTE: will throw error if use get_shape()
    # value = tf.squeeze(value)

    # quantize
    value = tf.round(value * 255)
    indices = tf.cast(value, np.int32)

    # gather
    colormap = cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = colormap(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    value = tf.cast(value * 255, tf.uint8)
    return value
####
def make_image(x, cy, cx, scale_y, scale_x):
    """
    Take 1st image from x and turn channels representations
    into 2D image, with cx number of channels in x-axis and
    cy number of channels in y-axis
    """
    # norm x for better visual
    x = tf.transpose(x,(0,2,3,1)) # NHWC
    max_x = tf.reduce_max(x, axis=-1, keep_dims=True)
    min_x = tf.reduce_min(x, axis=-1, keep_dims=True)
    x = 255 * (x - min_x) / (max_x - min_x)
    ###
    x_shape = tf.shape(x)
    channels = x_shape[-1]
    iy , ix = x_shape[1], x_shape[2] 
    ###
    x = tf.slice(x,(0,0,0,0),(1,-1,-1,-1))
    x = tf.reshape(x,(iy,ix,channels))
    ix += 4
    iy += 4
    x = tf.image.resize_image_with_crop_or_pad(x, iy, ix)
    x = tf.reshape(x,(iy,ix,cy,cx)) 
    x = tf.transpose(x,(2,0,3,1)) #cy,iy,cx,ix
    x = tf.reshape(x,(1,cy*iy,cx*ix,1))
    x = resize_op(x, scale_y, scale_x)
    return tf.cast(x, tf.uint8)
####
