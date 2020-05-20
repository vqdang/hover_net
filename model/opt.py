import torch.optim as optim

from run_utils.callbacks.base import *
from run_utils.callbacks.logging import *
from run_utils.engine import Events

from .run_step import train_step, valid_step

from model.net_desc import HoVerNet


# TODO: training config only ?
# TODO: switch all to function name String for all option
train_config = {

    #------------------------------------------------------------------

    # ! All phases have the same number of run engine
    # phases are run sequentially from index 0 to N
    'phase_list': [
        {
            'run_info': {
                # may need more dynamic for each network
                'net': {
                    'desc': lambda: HoVerNet(3, freeze=True),
                    'optimizer': [
                        optim.Adam,
                        {  # should match keyword for parameters within the optimizer
                            'lr': 1.0e-4,  # initial learning rate,
                            'betas': (0.9, 0.999)
                        },
                    ],
                    # learning rate scheduler
                    'lr_scheduler': lambda x: optim.lr_scheduler.StepLR(x, 25),

                    # path to load, -1 to auto load checkpoint from previous phase,
                    # None to start from scratch
                    'pretrained': 'ImageNet-ResNet50-Preact-Pytorch.npz',
                },
            },

            'nr_epochs': 50,
        },

        {
            'run_info': {
                # may need more dynamic for each network
                'net': {
                    'desc': lambda: HoVerNet(3, freeze=False),
                    'optimizer': [
                        optim.Adam,
                        {  # should match keyword for parameters within the optimizer
                            'lr': 1.0e-4,  # initial learning rate,
                            'betas': (0.9, 0.999)
                        },
                    ],
                    # learning rate scheduler
                    'lr_scheduler': lambda x: optim.lr_scheduler.StepLR(x, 25),

                    # path to load, -1 to auto load checkpoint from previous phase,
                    # None to start from scratch
                    'pretrained': -1,
                },
            },

            'nr_epochs': 20,
        }
    ],

    #------------------------------------------------------------------

    # TODO: dynamically for dataset plugin selection and processing also?
    # all enclosed engine shares the same neural networks
    # as the on at the outer calling it
    'run_engine': {
        'train': {
            # TODO: align here, file path or what? what about CV?
            'dataset': '',  # whats about compound dataset ?
            'nr_procs': 4,  # number of threads for dataloader

            'batch_size': 4,
            'run_step': train_step,  # TODO: function name or function variable ?
            'reset_per_run': False,

            # callbacks are run according to the list order of the event
            'callbacks': {
                Events.STEP_COMPLETED: [
                    # LoggingGradient(),
                    ScalarMovingAverage(),
                ],
                Events.EPOCH_COMPLETED: [
                    TrackLr(),
                    CheckpointSaver(),
                    VisualizeOutput(),
                    LoggingEpochOutput(),
                    TriggerEngine('valid'),
                    ScheduleLr(),
                ]
            },
        },
        'valid': {
            'dataset': '',  # whats about compound dataset ?
            'nr_procs': 0,  # number of threads for dataloader

            'batch_size': 4,
            'run_step': valid_step,
            'reset_per_run': True,  # * to stop aggregating output etc. from last run

            # callbacks are run according to the list order of the event
            'callbacks': {
                Events.STEP_COMPLETED: [
                    AccumulateRawOutput(),
                ],
                Events.EPOCH_COMPLETED: [
                    ProcessAccumulatedRawOutput(),
                    LoggingEpochOutput(),
                ]
            },
        },
    },
}
