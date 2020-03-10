
import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorpack import Inferencer, logger
from tensorpack.callbacks import (DataParallelInferenceRunner, ModelSaver,
                                  MinSaver, MaxSaver, ScheduledHyperParamSetter)
from tensorpack.tfutils import SaverRestore, get_model_loader
from tensorpack.train import (SyncMultiGPUTrainerParameterServer, TrainConfig,
                              launch_train_with_config)

import loader.loader as loader
from config import Config
from misc.utils import get_files

import matplotlib.pyplot as plt

class StatCollector(Inferencer, Config):
    """
    Accumulate output of inference during training.
    After the inference finishes, calculate the statistics
    """
    def __init__(self, prefix='valid'):
        super(StatCollector, self).__init__()
        self.prefix = prefix

    def _get_fetches(self):
        return self.train_inf_output_tensor_names

    def _before_inference(self):
        self.true_list = []
        self.pred_list = []

    def _on_fetches(self, outputs):
        pred, true = outputs
        self.true_list.extend(true)
        self.pred_list.extend(pred)
 
    def _after_inference(self):
        # ! factor this out
        def _dice(true, pred, label):
            true = np.array(true == label, np.int32)            
            pred = np.array(pred == label, np.int32)            
            inter = (pred * true).sum()
            total = (pred + true).sum()
            return 2 * inter /  (total + 1.0e-8)

        stat_dict = {}
        pred = np.array(self.pred_list)
        true = np.array(self.true_list)

        # have to get total number pixels for mean per pixel
        nr_pixels = np.size(true[...,:1])

        if self.type_classification: 
         
            pred_type = pred[...,:self.nr_types]
            pred_inst = pred[...,self.nr_types:]

            true_inst = true
            true_type = true[...,1]
            true_np = (true_type > 0).astype('int32')
           
        else:
            pred_inst = pred
            true_inst = true
            true_np = true[...,0]

        # * index selection followed what is defined in the graph
        # * and all model's graphs must follow same index ordering protocol

        # classification statistic
        if self.model_type == 'dist':
            # regression
            pred_dst = pred_inst[...,-1]
            true_dst = true_inst[...,-1]
            error = pred_dst - true_dst
            mse = np.sum(error * error) / nr_pixels
            stat_dict[self.prefix + '_mse'] = mse
        elif self.model_type == 'np_hv':
            pred_hv = pred_inst[...,-2:]
            true_hv = true_inst[...,-2:]
            error = pred_hv - true_hv
            mse = np.sum(error * error) / nr_pixels
            stat_dict[self.prefix + '_mse'] = mse

        # classification statistic
        if self.model_type != 'dist':
            pred_np = pred_inst[...,0]
            true_np = true_inst[...,0]

            pred_np[pred_np >  0.5] = 1.0
            pred_np[pred_np <= 0.5] = 0.0

            accuracy = (pred_np == true_np).sum() / nr_pixels
            inter = (pred_np * true_np).sum()
            total = (pred_np + true_np).sum()
            dice = 2 * inter / (total + 1.0e-8)

            stat_dict[self.prefix + '_acc' ] = accuracy
            stat_dict[self.prefix + '_dice'] = dice

            if self.model_type == 'dcan':
                # do one more for contour
                pred_np = pred_inst[...,1]
                true_np = true_inst[...,1]
                pred_np[pred_np >  0.5] = 1.0
                pred_np[pred_np <= 0.5] = 0.0

                inter = (pred_np * true_np).sum()
                total = (pred_np + true_np).sum()
                dice = 2 * inter / (total + 1.0e-8)

                stat_dict[self.prefix + '_cnt_dice'] = dice

        if self.type_classification:
            pred_type = np.argmax(pred_type, axis=-1)

            type_dict = self.nuclei_type_dict
            type_dice_list = []
            for type_name, type_id in type_dict.items():
                dice_val = _dice(true_type, pred_type, type_id)
                type_dice_list.append(dice_val)
                stat_dict['%s_dice_%s' % (self.prefix, type_name)] = dice_val

        return stat_dict
####

###########################################
class Trainer(Config):   
    ####
    def get_datagen(self, batch_size, mode='train', view=False):
        if mode == 'train':
            augmentors = self.get_train_augmentors(
                                            self.train_input_shape,
                                            self.train_mask_shape,
                                            view)
            data_files = get_files(self.train_dir, self.data_ext)
            data_generator = loader.train_generator
            nr_procs = self.nr_procs_train
        else:
            augmentors = self.get_valid_augmentors(
                                            self.infer_input_shape,
                                            self.infer_mask_shape,
                                            view)
            data_files = get_files(self.valid_dir, self.data_ext)
            data_generator = loader.valid_generator
            nr_procs = self.nr_procs_valid

        # set nr_proc=1 for viewing to ensure clean ctrl-z
        nr_procs = 1 if view else nr_procs
        dataset = loader.DatasetSerial(data_files)
        datagen = data_generator(dataset,
                        shape_aug=augmentors[0],
                        input_aug=augmentors[1],
                        label_aug=augmentors[2],
                        batch_size=batch_size,
                        nr_procs=nr_procs)
        
        return datagen      
    ####
    def view_dataset(self, mode='train'):
        assert mode == 'train' or mode == 'valid', "Invalid view mode"
        datagen = self.get_datagen(4, mode=mode, view=True)
        loader.visualize(datagen, 4)
        return
    ####
    def run_once(self, opt, sess_init=None, save_dir=None):
        ####
        train_datagen = self.get_datagen(opt['train_batch_size'], mode='train')
        valid_datagen = self.get_datagen(opt['infer_batch_size'], mode='valid')

        ###### must be called before ModelSaver
        if save_dir is None:
            logger.set_logger_dir(self.save_dir)
        else:
            logger.set_logger_dir(save_dir)

        ######            
        model_flags = opt['model_flags']
        model = self.get_model()(**model_flags)
        ######
        callbacks=[
                ModelSaver(max_to_keep=opt['nr_epochs']),
        ]

        for param_name, param_info in opt['manual_parameters'].items():
            model.add_manual_variable(param_name, param_info[0])
            callbacks.append(ScheduledHyperParamSetter(param_name, param_info[1]))
        # multi-GPU inference (with mandatory queue prefetch)
        infs = [StatCollector()]
        callbacks.append(DataParallelInferenceRunner(
                                valid_datagen, infs, list(range(nr_gpus))))
        callbacks.append(MaxSaver('valid_dice'))
        
        ######
        steps_per_epoch = train_datagen.size() // nr_gpus

        config = TrainConfig(
                    model           = model,
                    callbacks       = callbacks      ,
                    dataflow        = train_datagen  ,
                    steps_per_epoch = steps_per_epoch,
                    max_epoch       = opt['nr_epochs'],
                )
        config.session_init = sess_init

        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpus))
        tf.reset_default_graph() # remove the entire graph in case of multiple runs
        return
    ####
    def run(self):
        def get_last_chkpt_path(prev_phase_dir):
            stat_file_path = prev_phase_dir + '/stats.json'
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            chkpt_list = [epoch_stat['global_step'] for epoch_stat in info]
            last_chkpts_path = "%smodel-%d.index" % (prev_phase_dir, max(chkpt_list))
            return last_chkpts_path

        phase_opts = self.training_phase

        if len(phase_opts) > 1:
            for idx, opt in enumerate(phase_opts):
                random.seed(self.seed)
                np.random.seed(self.seed)
                tf.random.set_random_seed(self.seed)

                log_dir = '%s/%02d/' % (self.save_dir, idx)
                pretrained_path = opt['pretrained_path'] 
                if pretrained_path == -1:
                    pretrained_path = get_last_chkpt_path(prev_log_dir)
                    init_weights = SaverRestore(pretrained_path, ignore=['learning_rate'])
                elif pretrained_path is not None:
                    init_weights = get_model_loader(pretrained_path)
                self.run_once(opt, sess_init=init_weights, save_dir=log_dir)
                prev_log_dir = log_dir
        else:
            random.seed(self.seed)
            np.random.seed(self.seed)
            tf.random.set_random_seed(self.seed)

            opt = phase_opts[0]
            init_weights = None
            if 'pretrained_path' in opt:
                assert opt['pretrained_path'] != -1
                init_weights = get_model_loader(opt['pretrained_path'])
            self.run_once(opt, sess_init=init_weights, save_dir=self.save_dir)

        return
    ####
####

###########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help="comma separated list of GPU(s) to use.")
    parser.add_argument('--view', help="view dataset, received either 'train' or 'valid' as input")
    args = parser.parse_args()

    trainer = Trainer()
    if args.view:
        trainer.view_dataset(args.view)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        nr_gpus = len(args.gpu.split(','))
        trainer.run()
