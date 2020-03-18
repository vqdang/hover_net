import os
import random
import shutil

import numpy as np
import torch
from imgaug import imgaug as ia
from termcolor import colored


####
def check_manual_seed(seed):
    """ 
    If manual seed is not specified, choose a random one and communicate it to the user.
    """

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # ia.random.seed(seed)

    print('Using manual seed: {seed}'.format(seed=seed))
    return

####
def check_log_dir(log_dir):
    # check if log dir exist
    if os.path.isdir(log_dir):
        color_word = colored('WARNING', color='red', attrs=['bold', 'blink'])
        print('%s: %s exist!' % (color_word, colored(log_dir, attrs=['underline'])))
        while (True):
            print('Select Action: d (delete) / q (quit)', end='')
            key = input()
            if key == 'd':
                shutil.rmtree(log_dir)
                break
            elif key == 'q':
                exit()
            else:
                color_word = colored('ERR', color='red')
                print('---[%s] Unrecognize Characters!' % color_word)
    return
