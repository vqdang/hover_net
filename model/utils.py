import math
import numpy as np

import torch
import torch.nn.functional as F

from matplotlib import cm


####
def crop_op(x, cropping, data_format='NCHW'):
    """
    Center crop image

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == 'NCHW':
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


####
def crop_to_shape(x, y, data_format='NCHW'):
    """
    Centre crop x so that x has shape of y

    y dims must be smaller than x dims
    """

    assert y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1], \
        'Ensure that y dimensions are smaller than x dimensions!'

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
def xentropy_loss(pred, true, **kwargs):
    """
    Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss
    """
    pred = pred.permute(0, 3, 1, 2)
    return F.cross_entropy(pred, true, **kwargs)


####
def dice_loss(output, target, loss_type='sorensen', smooth=1e-3):
    """ TODO: modify docstring in line with others
    Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
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
    """
    Calculate mean squared error loss

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps 
    
    Returns:
        loss: mean squared error
    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss


####
def msge_loss(true, pred, focus):
    """
    Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients
    """
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
        """
        For calculating gradient
        """
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = torch.tensor(kernel_h, requires_grad=False)  # constant
        kernel_v = torch.tensor(kernel_v, requires_grad=False)  # constant
        kernel_h = kernel_h.view(1, 1, 5, 5).to('cuda')  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5).to('cuda')  # constant

        h_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([dh_ch, dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1)  # to NHWC
        return dhv

    focus = focus[..., None]  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    pred_grad = get_gradient_hv(pred)
    true_grad = get_gradient_hv(true)
    loss = pred_grad - true_grad
    loss = focus.float() * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss
