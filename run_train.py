
import cv2

cv2.setNumThreads(0)
import inspect
import logging
import os
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
# TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader

from run_utils.engine import RunEngine
from run_utils.utils import (
    check_manual_seed, colored,
    convert_pytorch_checkpoint
)


# * must initialize augmentor per worker, else duplicated
# * rng generators may happen
def worker_init_fn(worker_id):
    # ! to make the seed chain reproducible, must use the torch random, not numpy
    # the torch rng from main thread will regenerate a base seed, which is then
    # copied into the dataloader each time it created (i.e start of each epoch)
    # then dataloader with this seed will spawn worker,
    # now we reseed the worker
    worker_info = torch.utils.data.get_worker_info()
    # to make it more random, simply switch torch.randint to np.randint
    worker_seed = torch.randint(0, 2**32, (1,))[0].cpu().item() + worker_id
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    # retrieve the dataset copied into this worker process
    # then set the random seed for each augmentation
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return


class RunManager(object):
    """
    Either used to view the dataset or
    to initialise the main training loop. 
    """
    def __init__(self, **kwargs):
        self.phase_idx = 0
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        return

    ####
    def _get_datagen(self, batch_size, run_mode, subset_name, nr_procs=0):
        nr_procs = nr_procs if not self.debug else 0

        input_dataset = self.create_dataset(
                                run_mode=run_mode,
                                subset_name=subset_name,
                                setup_augmentor=nr_procs == 0)
        logging.info(
            f'Dataset {run_mode} - {subset_name} : {len(input_dataset)}')

        dataloader = DataLoader(
                        input_dataset,
                        num_workers=nr_procs,
                        batch_size=batch_size,
                        shuffle=run_mode == 'train',
                        drop_last=run_mode == 'train',
                        worker_init_fn=worker_init_fn,
                    )
        return dataloader

    ####
    def _run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None):
        """
        Simply run the defined run_step of the related method once
        """
        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            # check_log_dir(log_dir)
            # rm_n_mkdir(log_dir)
            import joblib
            tfwriter = SummaryWriter(log_dir=log_dir)
            log_file = log_dir + "/stats.dat"
            joblib.dump({}, log_file)
            log_info = {
                "log_file": log_file,
                "tfwriter": tfwriter,
            }

        # ! create list of data loader
        def create_loader_dict(run_mode, loader_name_list):
            loader_dict = {}
            for loader_name in loader_name_list:
                loader_opt = opt['loader'][loader_name]
                loader_dict[loader_name] = self._get_datagen(
                        loader_opt['batch_size'],
                        run_mode, loader_name,
                        nr_procs=loader_opt['nr_procs'])
            return loader_dict

        ####
        def get_last_chkpt_path(prev_phase_dir, net_name):
            info = joblib.load(f'{prev_phase_dir}/stats.dat')
            # ! prioritize epoch over step if both exist
            epoch_list = [int(v) for v in info.keys()]
            last_chkpts_path = (
                f"{prev_phase_dir}/"
                f"{net_name}_epoch={max(epoch_list):2d}.tar"
            )
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

            pretrained_path = net_info["pretrained"]
            if pretrained_path is not None:
                if pretrained_path == -1:
                    # * depend on logging format so may be
                    # * broken if logging format has been changed
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    net_state_dict = torch.load(pretrained_path)["desc"]
                else:
                    chkpt_ext = os.path.basename(pretrained_path).split(".")[-1]
                    if chkpt_ext == "npz":
                        net_state_dict = dict(np.load(pretrained_path))
                        net_state_dict = {
                            k: torch.from_numpy(v)
                            for k, v in net_state_dict.items()
                        }
                    elif chkpt_ext == "tar":  # ! assume same saving format we desire
                        net_state_dict = torch.load(pretrained_path)["desc"]

                colored_word = colored(net_name, color="red", attrs=["bold"])
                logging.info(
                    f"Model `{colored_word}` pretrained path: {pretrained_path}"
                )

                # load_state_dict returns (missing keys, unexpected keys)
                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # * uncomment for your convenience
                logging.info(f"Missing Variables: {load_feedback[0]}")
                logging.info(f"Detected Unknown Variables: {load_feedback[1]}")

            # net_desc = torch.jit.script(net_desc)
            net_desc = DataParallel(net_desc)
            net_desc = net_desc.to('cuda')
            # print(net_desc) # * dump network definition or not?
            optimizer, optimizer_args = net_info['optimizer']
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            # TODO: expand for external aug for scheduler
            nr_iter = opt['nr_epochs']
            scheduler = net_info['lr_scheduler'](optimizer, nr_iter)
            net_run_info[net_name] = {
                'desc': net_desc,
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                # TODO: standardize API for external hooks
                'extra_info': net_info['extra_info']
            }

        # parsing the running engine configuration
        assert 'train' in run_engine_opt, \
            'No engine for training detected in description file'

        # initialize runner and attach callback afterward
        # * all engine shared the same network info declaration
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            runner_loader_dict = create_loader_dict(
                runner_name, runner_opt['loader'])
            runner_dict[runner_name] = RunEngine(
                loader_dict=runner_loader_dict,
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
                        triggered_runner_name = (
                            callback.triggered_engine_name
                        )
                        callback.triggered_engine = (
                            runner_dict[triggered_runner_name]
                        )
                    runner.add_event_handler(event, callback)

        # retrieve main runner
        main_runner = runner_dict['train']
        main_runner.separate_loader_output = False
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        # start the run loop
        main_runner.run(opt['nr_epochs'])

        logging.info('\n')
        logging.info("#" * 16)
        logging.info('\n')
        return

    ####
    def run(self):
        """
        Define multi-stage run or cross-validation or whatever in here
        """
        phase_list = self.model_config['phase_list']
        engine_opt = self.model_config['run_engine']

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            if len(phase_list) == 1:
                save_path = self.log_dir
            else:
                save_path = self.log_dir + '/%02d' % (phase_idx)
            self._run_once(
                phase_info, engine_opt, save_path,
                prev_log_dir=prev_save_path)
            prev_save_path = save_path
            self.phase_idx += 1
