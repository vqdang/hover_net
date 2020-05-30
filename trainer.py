"""trainer.py

Main HoVer-Net training script.

Usage:
  trainer.py [--gpu=<id>] [--view=<dset>]
  trainer.py (-h | --help)
  trainer.py --version

Options:
  -h --help       Show this string.
  --version       Show version.
  --gpu=<id>      Comma separated GPU list.  
  --view=<dset>   Visualise images after augmentation. Choose 'train' or 'valid'.
"""


from docopt import docopt
import numpy as np
import matplotlib
import glob
import inspect
import importlib
import shutil
import argparse
import os
import json

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from run_utils.utils import check_log_dir, check_manual_seed, colored
from run_utils.engine import RunEngine

from config import Config
import dataset

from tensorboardX import SummaryWriter
from dataloader.loader import TrainSerialLoader


####
class Trainer(Config):
    """
    Either used to view the dataset or
    to initialise the main training loop. 
    """
    def __init__(self):
        super().__init__()
        self.model_config = self.model_config_file.__getattribute__('train_config')
        self.dataset_info = getattr(dataset, self.dataset_name)(self.type_classification)

    ####
    def view_dataset(self, mode='train'):
        check_manual_seed(self.seed)
        dataloader = self.get_datagen(1, mode)
        for batch_data in dataloader: # convert from Tensor to Numpy
            batch_data_np = {k : v.numpy() for k, v in batch_data.items()}
            TrainSerialLoader.view(batch_data_np)
        return

    ####
    def get_datagen(self, batch_size, run_mode, nr_procs=0, fold_idx=0):       
        # TODO: flag for debug mode

        # ! Hard assumption on file type
        file_list = []
        if run_mode == 'train':
            data_dir_list = self.dataset_info.train_dir_list
        else:
            data_dir_list = self.dataset_info.valid_dir_list
        for dir_path in data_dir_list:
            file_list.extend(glob.glob('%s/*.npy' % dir_path))
        file_list.sort() # to always ensure same input ordering
        assert len(file_list) > 0, \
                'No .npy found for `%s`, please check `%s` in `config.py`' %\
                (run_mode, '%s_dir_list' % run_mode)

        input_dataset = TrainSerialLoader(file_list, mode=run_mode, 
                                            **self.shape_info[run_mode])

        nr_procs =  nr_procs if not self.debug else 0
        dataloader = DataLoader(input_dataset, 
                        num_workers= nr_procs, 
                        batch_size = batch_size, 
                        shuffle    = run_mode=='train', 
                        drop_last  = run_mode=='train')
        return dataloader

    ####
    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0):
        """
        Simply run the defined run_step of the related method once
        """

        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            # check_log_dir(log_dir)
            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + '/stats.json'
            with open(json_log_file, 'w') as json_file:
                json.dump({}, json_file) # create empty file
            log_info = {
                'json_file' : json_log_file,
                'tfwriter'  : tfwriter,
            }

        ####
        def get_last_chkpt_path(prev_phase_dir, net_name):
            stat_file_path = prev_phase_dir + '/stats.json'
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            epoch_list = [int(v) for v in info.keys()]
            last_chkpts_path = "%s/%s_epoch=%d.tar" % (prev_phase_dir, 
                                        net_name, max(epoch_list))
            return last_chkpts_path

        # TODO: adding way to load pretrained weight or resume the training
        # parsing the network and optimizer information
        net_run_info = {}
        net_info_opt = opt['run_info']
        for net_name, net_info in net_info_opt.items():                   
            assert inspect.isclass(net_info['desc']) \
                        or inspect.isfunction(net_info['desc']), \
                "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info['desc']()

            # TODO: customize print-out for each run ?
            # summary_string(net_desc, (3, 270, 270), device='cpu')

            pretrained_path = net_info['pretrained']
            if pretrained_path is not None:
                if pretrained_path == -1:
                    # * depend on logging format so may be broken if logging format has been changed
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    net_state_dict = torch.load(pretrained_path)['desc']
                else:
                    net_state_dict = dict(np.load(pretrained_path))
                    net_state_dict = {k : torch.Tensor(v) for k, v in net_state_dict.items()}
                
                colored_word = colored(net_name, color='red', attrs=['bold'])
                print('Use pretrained path for %s: %s' % (colored_word, pretrained_path))

                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # load_state_dict return (missing keys, unexpected keys)

            # net_desc = DataParallel(net_desc)
            net_desc = net_desc.to('cuda')
            # print(net_desc) # * dump network definition or not?
            optimizer, optimizer_args = net_info['optimizer']
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            scheduler = net_info['lr_scheduler'](optimizer)
            net_run_info[net_name] = {
                'desc' : net_desc,
                'optimizer' : optimizer,
                'lr_scheduler' : scheduler,
            }

        # parsing the running engine configuration
        assert 'train' in run_engine_opt, 'No engine for training detected in description file'

        # initialize runner and attach callback afterward
        # * all engine shared the same network info declaration
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            # TODO: align naming protocol
            runner_dict[runner_name] = RunEngine(
                dataloader=self.get_datagen(runner_opt['batch_size'], 
                                    runner_name, nr_procs=runner_opt['nr_procs'],
                                    fold_idx=fold_idx),
                engine_name=runner_name,
                run_step=runner_opt['run_step'],
                run_info=net_run_info,
                log_info=log_info,
            )

        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]['callbacks']
            for event, callback_list, in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        # import tqdm
        # for i in range(0, 10):
        #     dataloader = runner_dict['train'].dataloader
        #     pbar = tqdm.tqdm(total=len(dataloader), leave=True, ascii=True)
        #     for data_batch in dataloader:
        #         pbar.update()
        #     pbar.close()

        #     dataloader = runner_dict['valid'].dataloader
        #     pbar = tqdm.tqdm(total=len(dataloader), leave=True, ascii=True)
        #     for data_batch in dataloader:
        #         pbar.update()
        #     pbar.close()

        # retrieve main runner
        main_runner = runner_dict['train']
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        # start the run loop 
        main_runner.run(opt['nr_epochs'])

        print('\n')
        print('########################################################')
        print('########################################################')
        print('\n')
        return
    
    ####
    def run(self):
        """
        Define multi-stage run or cross-valid or whatever in here
        """

        phase_list = self.model_config['phase_list']
        engine_opt = self.model_config['run_engine']

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            save_path = self.log_dir + '/%02d' % (phase_idx)
            self.run_once(phase_info, engine_opt, save_path, prev_log_dir=prev_save_path)
            prev_save_path = save_path


####
if __name__ == '__main__':
    args = docopt(__doc__, version='HoVer-Net v1.0')
    trainer = Trainer()

    if args['--view'] and args['--gpu']:
        raise Exception(
            'Supply only one of --view and --gpu.')

    if args['--view']:
        if args['--view'] != 'train' and args['--view'] != 'valid':
            raise Exception(
                'Use "train" or "valid" for --view.')
        trainer.view_dataset(args['--view'])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        # nr_gpus = len(args['--gpu'].split(','))
        trainer.run()
