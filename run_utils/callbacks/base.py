
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mode as major_value
from sklearn.metrics import confusion_matrix

from misc.utils import center_pad_to_shape

####
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
        super(TrackLr, self).__init__()
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
        super(ScheduleLr, self).__init__()
        
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
        if not state.logging: return
        for net_name, net_info in state.run_info.items():
            net_checkpoint = {key : module.state_dict() \
                        for key, module in net_info.items()}
            torch.save(net_checkpoint, '%s/%s_epoch=%d.tar' %\
                    (state.log_dir, net_name, state.curr_epoch))
        return

####
class AccumulateOutput(BaseCallbacks):
    def run(self, state, event):
        step_output = state.step_raw_output
        accumulated_output = state.epoch_accumulated_output

        for key, step_value in step_output.items():
            if key in accumulated_output:
                accumulated_output[key].extend(list(step_value))
            else:
                accumulated_output[key] = list(step_value)
        return

####
class ProcessAccumulatedOutput(BaseCallbacks):
    def __init__(self, per_n_epoch=1):
        super(ProcessAccumulatedOutput, self).__init__()
        self.per_n_epoch = per_n_epoch

    def run(self, state, event):
        current_epoch = state.curr_epoch
        # if current_epoch % self.per_n_epoch != 0: return
        output = state.epoch_accumulated_output

        prob = np.array(output['prob'])
        true = np.array(output['true'])
        # threshold then get accuracy
        pred = np.argmax(prob, axis=-1)
        acc = np.mean(pred == true)

        # confusion matrix
        # * assume labels within GT are continuous integer
        conf_mat = confusion_matrix(true, pred, 
                    labels=np.arange(np.max(true)+1))
        #
        state.tracked_step_output['scalar']['acc'] = acc
        state.tracked_step_output['conf_mat']['conf_mat'] = conf_mat
        # else:
            # majority voting to get the final prediction
        # pred = [] 
        # prev_epoch_true = np.array(state.epoch_accumulated_output['true'])
        # for epoch_output in state.run_accumulated_output:
        #     epoch_prob = np.array(epoch_output['prob'])
        #     epoch_pred = np.argmax(epoch_prob, axis=-1)
        #     epoch_true = np.array(epoch_output['true'])
        #     assert (epoch_true == prev_epoch_true).all()
        #     pred.append(epoch_pred)
        # pred = np.stack(pred, axis=-1)
        # pred = major_value(pred, axis=-1)[0]
        # true = np.array(state.epoch_accumulated_output['true'])
        # acc = np.mean(pred == true)

        # confusion matrix
        # # * assume labels within GT are continuous integer
        # conf_mat = confusion_matrix(true, pred, 
        #             labels=np.arange(np.max(true)+1))
        # #
        # state.tracked_step_output['scalar']['acc'] = acc
        # state.tracked_step_output['conf_mat']['conf_mat'] = conf_mat
        return

####
class ScalarMovingAverage(BaseCallbacks):
    """
    Calculate the running average for all scalar output of 
    each runstep of the attached RunEngine
    """
    def __init__(self, alpha=0.95):
        super(ScalarMovingAverage, self).__init__()
        self.alpha = alpha
        self.tracking_dict = {}

    def run(self, state, event):       
        step_output = state.step_raw_output['EMA']

        for key, current_value in step_output.items():
            if key in self.tracking_dict:
                old_ema_value = self.tracking_dict[key]
                # calculate the exponential moving average
                new_ema_value = old_ema_value * self.alpha  \
                                + (1.0 - self.alpha) * current_value
                self.tracking_dict[key] = new_ema_value
            else: # init for variable which appear for the first time
                new_ema_value = current_value 
                self.tracking_dict[key] = new_ema_value

        state.tracked_step_output['scalar'] = self.tracking_dict
        return

####
