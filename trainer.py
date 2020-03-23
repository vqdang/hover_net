

import matplotlib
# * syn where to set this
# must use 'Agg' to plot out onto image
# matplotlib.use('Agg') 

import glob
import inspect
import importlib
import shutil
import argparse
import os
import json

from torch.nn import DataParallel
from torch.utils.data import DataLoader

from run_utils.utils import *
from run_utils.engine import RunEngine

from config import Config

from tensorboardX import SummaryWriter
from dataloader.loader import TrainSerialLoader

####
class Trainer(Config):
    def __init__(self):
        super(Trainer, self).__init__()
        # TODO add error checking here
        self.model_config = self.model_config_file.__getattribute__('train_config')
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
        data_dir_list = getattr(self, '%s_dir_list' % run_mode)
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
        Simply run the defined run_step of the related method 1 time
        """
        device = 'cuda'
        check_manual_seed(self.seed)

        ####
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

        # * depend on logging format so may be broken if logging format has been changed
        def get_last_chkpt_path(prev_phase_dir, net_name):
            stat_file_path = prev_phase_dir + '/stats.json'
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            last_chkpts_path = "%s/%s_epoch=%s.tar" % (prev_phase_dir, 
                                        net_name, max(info.keys()))
            return last_chkpts_path

        ####
        # TODO: adding way to load preraining weight or resume the training
        # parsing the network and optimizer information
        net_run_info = {}
        net_info_opt = opt['run_info']
        for net_name, net_info in net_info_opt.items():                   
            assert inspect.isclass(net_info['desc']) \
                        or inspect.isfunction(net_info['desc']), \
                "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info['desc']()

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

            net_desc = DataParallel(net_desc).to('cuda')
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

        # retrieve main runner
        main_runner = runner_dict['train']
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        # finally start the run loop
        main_runner.run(opt['nr_epochs'])
        # print(net_desc.module.classifier.weight)

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
            save_path = 'exp/dump/%02d' % (phase_idx)
            self.run_once(phase_info, engine_opt, save_path, prev_log_dir=prev_save_path)
            prev_save_path = save_path
    ####

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', nargs='?', default="0,1", type=str,
                                help='comma separated list of GPU(s) to use.')
    parser.add_argument('--view', help='view dataset', action='store_true')
    args = parser.parse_args()      

    # ! fix this !!!!
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    trainer = Trainer()
    # trainer.view_dataset()
    trainer.run()
    # if args.view:
    #     trainer.view_dataset()
    #     exit()
    # else:
    #     trainer.run()
