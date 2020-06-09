import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss

####
def train_step(batch_data, run_info):
    # TODO: synchronize the attach protocol
    loss_func_dict = {
        'bce'  : xentropy_loss,
        'dice' : dice_loss,
        'mse'  : mse_loss,
        'msge' : msge_loss,
    }
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {'EMA' : {}} 
    track_value = lambda name, value: result_dict['EMA'].update({name: value})

    ####
    imgs = batch_data['img']
    true_np = batch_data['np_map']
    true_hv = batch_data['hv_map']
   
    imgs = imgs.to('cuda').type(torch.float32) # to NCHW
    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = torch.squeeze(true_np).to('cuda').type(torch.int64)
    true_hv = torch.squeeze(true_hv).to('cuda').type(torch.float32)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        'np' : true_np_onehot,
        'hv' : true_hv,
    }

    ####
    model     = run_info['net']['desc']
    optimizer = run_info['net']['optimizer']
    ####
    model.train() 
    model.zero_grad() # not rnn so not accumulate

    pred_dict = model(imgs)
    pred_dict = {k : v.permute(0, 2, 3 ,1).contiguous() for k, v in pred_dict.items()}
    pred_dict['np'] = F.softmax(pred_dict['np'], dim=-1) 

    ####
    loss = 0
    loss_opts = run_info['net']['extra_info']['loss']
    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict[branch_name], pred_dict[branch_name]]
            if loss_name == 'msge':
                loss_args.append(true_np_onehot[...,1])
            term_loss = loss_func(*loss_args)
            track_value('loss_%s_%s' % (branch_name, loss_name), 
                         term_loss.cpu().item())
            loss += loss_weight * term_loss

    track_value('overall_loss', loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    ####

    # pick 2 random sample from the batch for visualization
    sample_indices = torch.randint(0, true_np.shape[0], (2,))

    imgs = (imgs[sample_indices]).byte() # to uint8
    imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    pred_dict['np'] = pred_dict['np'][...,1] # return pos only
    pred_dict = {k : v[sample_indices].detach().cpu().numpy() for k, v in pred_dict.items()}

    true_dict['np'] = true_np
    true_dict = {k : v[sample_indices].detach().cpu().numpy() for k, v in true_dict.items()}
    
    # * Its up to user to define the protocol to process the raw output per step!
    result_dict['raw'] = { # protocol for contents exchange within `raw`
        'img': imgs,
        'np' : (true_dict['np'], pred_dict['np']),
        'hv' : (true_dict['hv'], pred_dict['hv'])
    }
    return result_dict

####
def valid_step(batch_data, run_info):

    ####
    imgs = batch_data['img']
    true_np = batch_data['np_map']
    true_hv = batch_data['hv_map']
   
    imgs_gpu = imgs.to('cuda').type(torch.float32) # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = torch.squeeze(true_np).to('cuda').type(torch.int64)
    true_hv = torch.squeeze(true_hv).to('cuda').type(torch.float32)

    true_dict = {
        'np' : true_np,
        'hv' : true_hv,
    }
    ####
    model = run_info['net']['desc']
    model.eval() # infer mode

    # --------------------------------------------------------------
    with torch.no_grad(): # dont compute gradient
        pred_dict = model(imgs_gpu)
        pred_dict = {k : v.permute(0, 2, 3 ,1).contiguous() for k, v in pred_dict.items()}
        pred_dict['np'] = F.softmax(pred_dict['np'], dim=-1)[...,1] 

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = { # protocol for contents exchange within `raw`
        'raw': {
            'imgs' : imgs.numpy(),
            'true_np' : true_dict['np'].cpu().numpy(),
            'true_hv' : true_dict['hv'].cpu().numpy(),
            'prob_np' : pred_dict['np'].cpu().numpy(),
            'pred_hv' : pred_dict['hv'].cpu().numpy()
        }
    }
    return result_dict

####
def infer_step(batch_data, model):

    ####
    patch_imgs = batch_data
   
    patch_imgs_gpu = patch_imgs.to('cuda').type(torch.float32) # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    ####
    model.eval() # infer mode

    # --------------------------------------------------------------
    with torch.no_grad(): # dont compute gradient
        pred_dict = model(patch_imgs_gpu)
        pred_dict = {k : v.permute(0, 2, 3 ,1).contiguous() for k, v in pred_dict.items()}
        pred_dict['np'] = F.softmax(pred_dict['np'], dim=-1)[...,1] 


    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = { # protocol for contents exchange within `raw`
        'raw': {
            'prob_np' : pred_dict['np'].cpu().numpy(),
            'pred_hv' : pred_dict['hv'].cpu().numpy()
        }
    }
    return result_dict
    
####
def viz_step_output(raw_data):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data['img']
    true_np, pred_np = raw_data['np']
    true_hv, pred_hv = raw_data['hv']

    aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
    aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]

    cmap = plt.get_cmap('jet')

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype('float32'))
        ch[ch > vmax] = vmax # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[...,:3] * 255).astype('uint8')
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)

        true_viz_list = [img]
        # cmap may randomly fails if of other types
        true_viz_list.append(colorize(true_np[idx], 0, 1))
        true_viz_list.append(colorize(true_hv[idx][...,0], -1, 1))
        true_viz_list.append(colorize(true_hv[idx][...,1], -1, 1))
        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img]
        # cmap may randomly fails if of other types
        pred_viz_list.append(colorize(pred_np[idx], 0, 1))
        pred_viz_list.append(colorize(pred_hv[idx][...,0], -1, 1))
        pred_viz_list.append(colorize(pred_hv[idx][...,1], -1, 1))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(
            np.concatenate([true_viz_list, pred_viz_list], axis=0)
        )
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list

####
from itertools import chain
def proc_valid_step_output(raw_data):
    # TODO: add auto populate from main state track list
    track_dict = {'scalar': {}, 'image' : {}}
    def track_value(name, value, vtype): return track_dict[vtype].update({name: value})

    def longlist2array(longlist):
        tmp = list(chain.from_iterable(longlist))
        return np.array(tmp).reshape((len(longlist),) + longlist[0].shape)

    # ! factor this out
    # def _dice(true, pred, label):
    #     true = np.array(true == label, np.int32)
    #     pred = np.array(pred == label, np.int32)
    #     inter = (pred * true).sum()
    #     total = (pred + true).sum()
    #     return 2 * inter / (total + 1.0e-8)

    # ! paging / caching problem when merging huge list ?
    # pred_np = longlist2array(raw_data['prob_np'])
    # true_np = longlist2array(raw_data['true_np'])
    # nr_pixels = np.size(true_np)
    # * NP segmentation statistic
    # pred_np[pred_np > 0.5]  = 1.0
    # pred_np[pred_np <= 0.5] = 0.0

    # acc_np = (pred_np == true_np).sum() / nr_pixels
    # dice_np = _dice(true_np, pred_np, 1)
    # track_value('np_acc', acc_np, 'scalar')
    # track_value('np_dice', dice_np, 'scalar')
    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    over_inter = 0
    over_total = 0
    over_correct = 0
    prob_np = raw_data['prob_np']
    true_np = raw_data['true_np']
    for idx in range(len(raw_data['true_np'])):
        patch_prob_np = prob_np[idx]
        patch_true_np = true_np[idx]
        patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
        inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
        correct = (patch_pred_np == patch_true_np).sum()
        over_inter += inter
        over_total += total
        over_correct += correct
    nr_pixels = len(true_np) * np.size(true_np[0])
    acc_np  = over_correct / nr_pixels
    dice_np = 2 * over_inter / (over_total + 1.0e-8)
    track_value('np_acc', acc_np, 'scalar')
    track_value('np_dice', dice_np, 'scalar')

    # * HV regression statistic
    pred_hv = raw_data['pred_hv']
    true_hv = raw_data['true_hv']

    over_squared_error = 0
    for idx in range(len(raw_data['true_np'])):
        patch_pred_hv = pred_hv[idx]
        patch_true_hv = true_hv[idx]
        squared_error = patch_pred_hv - patch_true_hv
        squared_error = squared_error * squared_error
        over_squared_error += squared_error.sum()
    mse = over_squared_error / nr_pixels
    track_value('hv_mse', mse, 'scalar')

    # pred_hv = longlist2array(raw_data['pred_hv'])
    # true_hv = longlist2array(raw_data['true_hv'])
    # error = pred_hv - true_hv
    # mse = np.sum(error * error) / nr_pixels
    # track_value('hv_mse', mse, 'scalar')

    # *
    imgs = raw_data['imgs']
    selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
    imgs    = np.array([imgs[idx] for idx in selected_idx])
    true_np = np.array([true_np[idx] for idx in selected_idx])
    true_hv = np.array([true_hv[idx] for idx in selected_idx])
    prob_np = np.array([prob_np[idx] for idx in selected_idx])
    pred_hv = np.array([pred_hv[idx] for idx in selected_idx])
    viz_raw_data = {
        'img': imgs,
        'np' : (true_np, prob_np),
        'hv' : (true_hv, pred_hv)
    }
    viz_fig = viz_step_output(viz_raw_data)
    track_dict['image']['output'] = viz_fig

    return track_dict
