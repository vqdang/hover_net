
import torch
import torch.nn.functional as F

from .utils import *

####
def train_step(batch, run_info):
    # TODO: synchronize the attach protocol
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {'EMA' : {}} 

    track_value = lambda name, value: result_dict['EMA'].update(name, value)

    ####
    batch_data, batch_label = batch
    batch_label = batch_label if len(batch_data) != 1 else batch_label[None]
    
    batch_data  = batch_data.float().to('cuda') # to NCHW
    batch_data = batch_data.permute(0, 3, 1, 2)

    batch_label = batch_label.to('cuda')
    true_np = batch_label[...,0]
    true_hv = batch_label[...,1:2]
    # true_tp = batch_label[...,2:]

    ####
    model     = run_info['net']['desc']
    optimizer = run_info['net']['optimizer']
    ####
    model.train() 
    model.zero_grad() # not rnn so not accumulate

    output_dict = model(batch_data)

    pred_np = output_dict['np'] # should be logit value, not softmax output
    pred_hv = output_dict['hv']

    prob_np = F.softmax(pred_np, dim=-1)
    ####
    loss = 0

    # TODO: adding loss weighting mechanism
    # * For Nulcei vs Background Segmentation 
    term_loss = F.cross_entropy(pred_np, true_np, reduction='mean')
    track_value('loss_xentropy_np', term_loss)
    loss += term_loss

    term_loss = dice_loss(pred_np[...,0], true_np[...,0]) \
              + dice_loss(pred_np[...,1], true_np[...,1])
    track_value('loss_dice_np', term_loss)
    loss += term_loss
    
    # * For Nulcei vs Background Segmentation 
    term_loss = mse_loss(pred_hv, true_hv)
    track_value('loss_mse_hv', term_loss)
    loss += term_loss

    focus_region = true_np[...,0]
    term_loss = msge_loss(pred_hv, true_hv, focus_region)
    track_value('loss_msge_hv', term_loss)
    loss += term_loss

    # gradient update
    loss.backward()
    optimizer.step()
    ####

    # * For visualizing thingy
    # orig_imgs = crop_op(orig_imgs, (190, 190), "NHWC")

    pred_np = colorize(prob_np[...,0], cmap='jet')
    true_np = colorize(true_np[...,0], cmap='jet')
    pred_h = colorize(pred_hv[...,0], vmin=-1, vmax=1, cmap='jet')
    pred_v = colorize(pred_hv[...,1], vmin=-1, vmax=1, cmap='jet')
    true_h = colorize(true_hv[...,0], vmin=-1, vmax=1, cmap='jet')
    true_v = colorize(true_hv[...,1], vmin=-1, vmax=1, cmap='jet')

    viz = torch.cat([pred_h, pred_v, pred_np, 
                     true_h, true_v, true_np], 2)
    viz = viz.cpu().numpy()

    import matplotlib.pyplot as plt

    return result_dict

####
def infer_step(batch, net):
    net.eval() # infer mode

    # batch is NHWC
    imgs = batch
    imgs = imgs.permute(0, 3, 1, 2) # to NCHW

    # push data to GPUs and convert to float32
    imgs = imgs.to('cuda').float()

    # -----------------------------------------------------------
    with torch.no_grad(): # dont compute gradient
        output_dict = net(imgs) # forward
        for output_name, output_value in output_dict.items():
            if output_name == 'np':                
                output_value = nn.functional.softmax(output_value, dim=1)
            output_dict[output_name] = output_value.permute(0, 2, 3, 1) # to NHWC
        output = torch.cat(list(output_dict.values()), -1)        
        output = output.cpu().numpy() 

    return output

