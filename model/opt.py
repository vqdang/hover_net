import torch.optim as optim

from run_utils.callbacks.base import * 
from run_utils.callbacks.logging import * 
from run_utils.engine import Events

from .desc import NetDesc
from .run_step import *

from torchvision.models.densenet import DenseNet


# TODO: training config only ?
# TODO: switch all to function name String for all option
train_config = {
    
    #------------------------------------------------------------------

    # ! All phases have the same number of run engine
    # phases are run sequentially from index 0 to N
    'phase_list' : [
        {
            'run_info' : {
                # may need more dynamic for each network
                'net' : {
                    'desc'       : lambda : NetDesc(3, freeze=False),
                    'optimizer'  : [
                        optim.Adam,
                        { # should match keyword for parameters within the optimizer
                            'lr'    : 1.0e-4, # initial learning rate,
                            'betas' : (0.5, 0.999)
                        },
                    ],
                    # learning rate scheduler
                    'lr_scheduler' : lambda x : optim.lr_scheduler.StepLR(x, 60),
                },
            },

            'nr_epochs' : 90, 
            # 'nr_epochs' : 1, 

            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained' : None,
        },

        {
            'run_info' : {
                # may need more dynamic for each network
                'net' : {
                    'desc'       : 'self',
                    'optimizer'  : [
                        'Adam',
                        { # should match keyword for parameters within the optimizer
                            'lr'    : 1.0e-3, # initial learning rate,
                            'betas' : (0.5, 0.999)
                        },
                    ],
                    # learning rate scheduler
                    'lr_scheduler' : ['StepLR', {'step_size' : 30, 'gamma' : 0.1}],

                    'freeze' : False, # additional flag for internal control rule
                },
            },

            'nr_epochs' : 90, 

            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained' : -1,
        }
    ],

    #------------------------------------------------------------------

    # TODO: dynamically for dataset plugin selection and processing also?
    # all enclosed engine shares the same neural networks
    # as the on at teh outer calling it
    'run_engine' : {
        'train' : {   
            # TODO: align here, file path or what? what about CV?
            'dataset'    : '', # whats about compound dataset ?
            'batch_size' : 16,
            'run_step'   : 'train_step',
            'reset_per_run' : False,

            # callbacks are run according to the list order of the event            
            'callbacks' : {
                Events.STEP_COMPLETED  : [
                    # LoggingGradient(),
                    ScalarMovingAverage(),
                ],
                Events.EPOCH_COMPLETED : [
                    TrackLr(),
                    CheckpointSaver(),
                    LoggingEpochOutput(),
                    TriggerEngine('valid'), 
                    ScheduleLr(),                  
                ]
            },
        },       
        'valid' : {
            'dataset'    : '', # whats about compound dataset ?
            'batch_size' : 16,
            'run_step'   : 'valid_step',
            'reset_per_run' : True, # * to stop aggregating output etc. from last run
            
            # callbacks are run according to the list order of the event            
            'callbacks' : {
                Events.STEP_COMPLETED  : [
                    AccumulateOutput(),
                ],
                Events.EPOCH_COMPLETED : [
                    ProcessAccumulatedOutput(),
                    LoggingEpochOutput(),
                ]
            },
        },
    }, 
}
