import math
import numpy as np

import torch
import torch.functional as F

from matplotlib import cm

####
def crop_op(x, cropping, data_format='NCHW'):
    """
    Center crop image
    Args:
        cropping is the substracted portion
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == 'NCHW':
        x = x[:,:,crop_t:-crop_b,crop_l:-crop_r]
    else:
        x = x[:,crop_t:-crop_b,crop_l:-crop_r,:]
    return x  

####
def crop_to_shape(x, y, data_format='NCHW'):
    """
    center cropping x so that x has shape of y

    y shape must be within x shape
    """
    x_shape = x.size()
    y_shape = y.size() 
    if data_format == 'NCHW':
        crop_shape = (x_shape[2] - y_shape[2], 
                      x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], 
                      x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)
####

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
def dice_loss(output, target, loss_type='sorensen', smooth=1e-3):
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
    target = torch.squeeze(target.float())
    output = torch.squeeze(output.float())

    inse = (output * target).sum()
    if loss_type == 'jaccard':
        l = (output * output).sum()
        r = (target * target).sum()
    elif loss_type == 'sorensen':
        l = output.sum()
        r = target.sum()
    else:
        raise Exception("Unknown loss_type")
    # already flatten
    dice = 1.0 - (2. * inse + smooth) / (l + r + smooth)
    ##
    return dice

####
def mse_loss(true, pred):
    ### regression loss
    loss = pred - true
    loss = (loss * loss).sum()
    return loss

####
def msge_loss(true, pred, focus):
    '''
    Assume channel 0 is Vertical and channel 1 is Horizontal
    '''
    ####
    def get_sobel_kernel(size):
        assert size % 2 == 1, 'Must be odd, get size=%d' % size

        h_range = np.arange(-size//2+1, size//2+1, dtype=np.float32)
        v_range = np.arange(-size//2+1, size//2+1, dtype=np.float32)
        h, v = np.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v   

    ####
    def get_gradient_hv(hv):
        ### for calculating gradient and the like
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = torch.tensor(kernel_h, requires_grad=False) # constant
        kernel_v = torch.tensor(kernel_v, requires_grad=False) # constant
        kernel_h = torch.view(5, 5, 1, 1) # constant
        kernel_v = torch.view(5, 5, 1, 1) # constant

        h_ch = hv[:,1].unsqueeze(1) # Nx1xHxW
        v_ch = hv[:,0].unsqueeze(1) # Nx1xHxW
        dh_ch = F.conv2d(h_ch, self.kernel_h, padding=2)
        dv_ch = F.conv2d(v_ch, self.kernel_v, padding=2)
        dhv = torch.cat([dh_ch, dv_ch], dim=1)
        return dhv

    focus = torch.cat([focus, focus], axis=-1)
    pred_grad = get_gradient_hv(pred)
    true_grad = get_gradient_hv(true) 
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focus region
    loss = loss / (focus.sum() + 1.0e-8)
    return loss

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
    if vmin is None: # TODO: untested
        # min over dim 1 and 2
        vmin = value.min(1).min(1)
        vmin = vmin.view(-1, 1, 1)
    if vmax is None: # TODO: untested
        vmax = value.max(1).max(1)
        vmax = vmin.view(-1, 1, 1)
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # quantize
    value = torch.round(value * 255)
    indices = value.int32()

    # gather
    colormap = cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = colormap(np.arange(256))[:, :3]
    colors = torch.from_numpy(colors).float()
    value = torch.gather(colors, indices)
    value = (value * 255).uint8()
    return value
####
