
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mode as major_value
from sklearn.metrics import confusion_matrix

from misc.utils import center_pad_to_shape, cropping_center


class BaseCallbacks(object):
    def __init__(self):
        self.engine_trigger = False

    def reset(self):
        pass

    def run(self, state, event):
        pass
####


class TrackLr(BaseCallbacks):
    """
    Add learning rate to tracking
    """

    def __init__(self, per_n_epoch=1, per_n_step=None):
        super().__init__()
        self.per_n_epoch = per_n_epoch
        self.per_n_step = per_n_step

    def run(self, state, event):
        # logging learning rate, decouple into another callback?
        run_info = state.run_info
        for net_name, net_info in run_info.items():
            lr = net_info['optimizer'].param_groups[0]['lr']
            state.tracked_step_output['scalar']['lr-%s' % net_name] = lr
        return
####


class ScheduleLr(BaseCallbacks):
    '''
    Trigger all scheduler
    '''

    def __init__(self):
        super().__init__()

    def run(self, state, event):
        # logging learning rate, decouple into another callback?
        run_info = state.run_info
        for net_name, net_info in run_info.items():
            net_info['lr_scheduler'].step()
        return
####


class TriggerEngine(BaseCallbacks):
    def __init__(self, triggered_engine_name, nr_epoch=1):
        self.engine_trigger = True
        self.triggered_engine_name = triggered_engine_name
        self.triggered_engine = None
        self.nr_epoch = nr_epoch

    def run(self, state, event):
        self.triggered_engine.run(chained=True,
                                  nr_epoch=self.nr_epoch,
                                  shared_state=state)
        return
####


class CheckpointSaver(BaseCallbacks):
    """
    Must declare save dir first in the shared global state of the
    attached engine
    """

    def run(self, state, event):
        if not state.logging:
            return
        for net_name, net_info in state.run_info.items():
            net_checkpoint = {key: module.state_dict()
                              for key, module in net_info.items()}
            torch.save(net_checkpoint, '%s/%s_epoch=%d.tar' %
                       (state.log_dir, net_name, state.curr_epoch))
        return
####


class AccumulateRawOutput(BaseCallbacks):
    def run(self, state, event):
        step_output = state.step_output['raw']
        accumulated_output = state.epoch_accumulated_output

        for key, step_value in step_output.items():
            if key in accumulated_output:
                accumulated_output[key].extend(list(step_value))
            else:
                accumulated_output[key] = list(step_value)
        return
####


class ProcessAccumulatedRawOutput(BaseCallbacks):
    def __init__(self, per_n_epoch=1):
        # TODO: allow dynamically attach specific procesing for `type`
        super().__init__()
        self.per_n_epoch = per_n_epoch

    def run(self, state, event):
        current_epoch = state.curr_epoch
        # if current_epoch % self.per_n_epoch != 0: return
        output_dict = state.epoch_accumulated_output

        # TODO: add auto populate from main state track list
        track_dict = {'scalar': {}}
        def track_value(name, value, vtype): return track_dict[vtype].update(
            {name: value})

        # ! factor this out
        def _dice(true, pred, label):
            true = np.array(true == label, np.int32)
            pred = np.array(pred == label, np.int32)
            inter = (pred * true).sum()
            total = (pred + true).sum()
            return 2 * inter / (total + 1.0e-8)

        pred_np = np.array(output_dict['prob_np'])
        true_np = np.array(output_dict['true_np'])
        nr_pixels = np.size(true_np)
        # * NP segmentation statistic
        pred_np[pred_np > 0.5] = 1.0
        pred_np[pred_np <= 0.5] = 0.0

        acc_np = (pred_np == true_np).sum() / nr_pixels
        dice_np = _dice(true_np, pred_np, 1)
        track_value('np_acc', acc_np, 'scalar')
        track_value('np_dice', dice_np, 'scalar')

        # * HV regression statistic
        pred_hv = np.array(output_dict['pred_hv'])
        true_hv = np.array(output_dict['true_hv'])
        error = pred_hv - true_hv
        mse = np.sum(error * error) / nr_pixels
        track_value('hv_mse', mse, 'scalar')

        # update global shared states
        state.tracked_step_output = track_dict
        return
####


class ScalarMovingAverage(BaseCallbacks):
    """
    Calculate the running average for all scalar output of 
    each runstep of the attached RunEngine
    """

    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha
        self.tracking_dict = {}

    def run(self, state, event):
        # TODO: protocol for dynamic key retrieval for EMA
        step_output = state.step_output['EMA']

        for key, current_value in step_output.items():
            if key in self.tracking_dict:
                old_ema_value = self.tracking_dict[key]
                # calculate the exponential moving average
                new_ema_value = old_ema_value * self.alpha  \
                    + (1.0 - self.alpha) * current_value
                self.tracking_dict[key] = new_ema_value
            else:  # init for variable which appear for the first time
                new_ema_value = current_value
                self.tracking_dict[key] = new_ema_value

        state.tracked_step_output['scalar'] = self.tracking_dict
        return
####


class VisualizeOutput(BaseCallbacks):
    def __init__(self, per_n_epoch=1):
        super().__init__()
        self.per_n_epoch = per_n_epoch

    def run(self, state, event):
        current_epoch = state.curr_epoch
        raw_output = state.step_output['raw']

        imgs = raw_output['img']
        true_np, pred_np = raw_output['np']
        true_hv, pred_hv = raw_output['hv']

        aligned_shape = [list(imgs.shape), list(
            true_np.shape), list(pred_np.shape)]
        aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]

        cmap = plt.get_cmap('jet')

        def colorize(ch, vmin, vmax):
            ch = np.squeeze(ch.astype('float32'))
            ch = ch / (vmax - vmin + 1.0e-16)
            # take RGB from RGBA heat map
            ch_cmap = (cmap(ch)[..., :3] * 255).astype('uint8')
            # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
            return ch_cmap

        viz_list = []
        for idx in range(2):
            # img = center_pad_to_shape(imgs[idx], aligned_shape)
            img = cropping_center(imgs[idx], aligned_shape)

            true_viz_list = [img]
            # cmap may randomly fails if of other types
            true_viz_list.append(colorize(true_np[idx], 0, 1))
            # map to [0,2] for better visualisation.
            # Note, [-1,1] is used for training.
            true_viz_list.append(colorize(true_hv[idx][..., 0] + 1, 0, 2))
            true_viz_list.append(colorize(true_hv[idx][..., 1] + 1, 0, 2))
            true_viz_list = np.concatenate(true_viz_list, axis=1)

            pred_viz_list = [img]
            # cmap may randomly fails if of other types
            pred_viz_list.append(colorize(pred_np[idx], 0, 1))
            # map to [0,2] for better visualisation.
            # Note, [-1,1] is used for training.
            pred_viz_list.append(colorize(pred_hv[idx][..., 0] + 1, 0, 2))
            pred_viz_list.append(colorize(pred_hv[idx][..., 1] + 1, 0, 2))
            pred_viz_list = np.concatenate(pred_viz_list, axis=1)

            viz_list.append(
                np.concatenate([true_viz_list, pred_viz_list], axis=0)
            )
        viz_list = np.concatenate(viz_list, axis=0)
        # plt.imshow(viz_list)
        # plt.savefig('dumpx.png', dpi=600)
        state.tracked_step_output['image']['output'] = viz_list
        return
