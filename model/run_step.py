import matplotlib.pyplot as plt
# * For visualizing thingy
import numpy as np
import torch
import torch.nn.functional as F

from misc.utils import center_pad_to_shape
from .utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss

####
def train_step(batch_data, run_info):
    # TODO: synchronize the attach protocol
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {'EMA' : {}} 
    track_value = lambda name, value: result_dict['EMA'].update({name: value})

    ####
    imgs = batch_data['img']
    true_np = batch_data['np_map']
    true_hv = batch_data['hv_map']
   
    imgs = imgs.float().to('cuda') # to NCHW
    imgs = imgs.permute(0, 3, 1, 2)

    # HWC
    true_np = torch.squeeze(true_np.long().to('cuda'))
    true_hv = torch.squeeze(true_hv.float().to('cuda'))

    # true_tp = batch_label[...,2:]

    ####
    model     = run_info['net']['desc']
    optimizer = run_info['net']['optimizer']
    ####
    model.train() 
    model.zero_grad() # not rnn so not accumulate

    output_dict = model(imgs)
    output_dict = {k : v.permute(0, 2, 3 ,1) for k, v in output_dict.items()}

    pred_np = output_dict['np'] # should be logit value, not softmax output
    pred_hv = output_dict['hv']

    prob_np = F.softmax(pred_np, dim=-1)
    ####
    loss = 0

    # TODO: adding loss weighting mechanism
    # * For Nulcei vs Background Segmentation 
    # NP branch

    # pred must be NCHW
    term_loss = xentropy_loss(pred_np, true_np, reduction='mean')
    track_value('np_xentropy_loss', term_loss.cpu().item())
    loss += term_loss

    true_np_onehot = F.one_hot(true_np) # need to dynamic for type later
    term_loss = dice_loss(prob_np[...,0], true_np_onehot[...,0]) \
              + dice_loss(prob_np[...,1], true_np_onehot[...,1])
    track_value('np_dice_loss', term_loss.cpu().item())
    loss += term_loss
    
    # HV branch
    term_loss = mse_loss(pred_hv, true_hv)
    track_value('hv_mse_loss', term_loss.cpu().item())
    loss += term_loss

    term_loss = msge_loss(pred_hv, true_hv, true_np)
    track_value('hv_msge_loss', term_loss.cpu().item())
    loss += term_loss

    track_value('overall_loss', loss.cpu().item())
    # * gradient update
    loss.backward()
    optimizer.step()
    ####

    # pick 2 random sample from the batch for visualization
    sample_indices = torch.randint(0, true_np.shape[0], (2,))

    imgs = (imgs[sample_indices]).byte() # to uint8
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()

    pred_hv = pred_hv.detach()[sample_indices].cpu().numpy()
    true_hv = true_hv[sample_indices].cpu().numpy()

    prob_np = pred_np.detach()[...,1:][sample_indices].cpu().numpy()
    true_np = true_np.float()[...,None][sample_indices].cpu().numpy()

    # plt.imshow(viz)
    # plt.savefig('dump.png', dpi=600)
    # exit()

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict['raw'] = { # protocol for contents exchange within `raw`
        'img': imgs,
        'np' : (true_np, prob_np),
        'hv' : (true_hv, pred_hv)
    }
    return result_dict

####
def valid_step(batch_data, run_info):

    ####
    imgs = batch_data['img']
    true_np = batch_data['np_map']
    true_hv = batch_data['hv_map']
   
    imgs = imgs.float().to('cuda') # to NCHW
    imgs = imgs.permute(0, 3, 1, 2)

    # HWC
    true_np = torch.squeeze(true_np.long().to('cuda'))
    true_hv = torch.squeeze(true_hv.float().to('cuda'))

    ####
    model = run_info['net']['desc']
    model.eval() # infer mode

    # -----------------------------------------------------------
    with torch.no_grad(): # dont compute gradient
        output_dict = model(imgs) # forward
    output_dict = {k : v.permute(0, 2, 3 ,1) for k, v in output_dict.items()}

    pred_np = output_dict['np'] # should be logit value, not softmax output
    pred_hv = output_dict['hv']
    prob_np = F.softmax(pred_np, dim=-1)[...,1]

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = { # protocol for contents exchange within `raw`
        'raw': {
            'true_np' : true_np.cpu().numpy(),
            'true_hv' : true_hv.cpu().numpy(),
            'prob_np' : prob_np.cpu().numpy(),
            'pred_hv' : pred_hv.cpu().numpy()
        }
    }
    return result_dict
