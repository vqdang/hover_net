import argparse
import collections
import importlib
import logging
import os

import joblib
import yaml

from dataloader.train_loader import FileLoader
from misc.utils import mkdir, recur_find_ext, rm_n_mkdir, rmdir


####
def load_yaml(path):
    with open(path) as fptr:
        info = yaml.full_load(fptr)
    return info


def update_nested_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update_nested_dict(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        # elif isinstance(val, list):
        #     orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=str, default='0,1')

    args = parser.parse_args()
    # print(args)

    logging.basicConfig(level=logging.INFO,)

    seed = 5
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    FOLD_IDX = 0
    WORKSPACE_DIR = 'exp_output/local'
    SAVE_ROOT = f'{WORKSPACE_DIR}/models/baseline/'

    splits = joblib.load('splits.dat')

    def run_one_split_with_param_set(save_path, split_info, param_kwargs):
        mkdir(save_path)

        template_paramset = load_yaml('param/template.yaml')

        # repopulate loader arg according to available subset info
        template_loader_kwargs = template_paramset['loader_kwargs']
        loader_kwargs = {
            k: template_loader_kwargs['train'] if 'train' in k else
            template_loader_kwargs['infer'] for k in split_info.keys()}
        template_paramset['loader_kwargs'] = loader_kwargs

        # * reset logger handler
        log_formatter = logging.Formatter(
            '|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d|%H:%M:%S'
        )
        log = logging.getLogger()  # root logger
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        new_hdlr_list = [
            logging.FileHandler(f"{save_path}/debug.log"),
            logging.StreamHandler()
        ]
        for hdlr in new_hdlr_list:
            hdlr.setFormatter(log_formatter)
            log.addHandler(hdlr)
        #

        train_loader_list = [
            v for v in split_info.keys() if 'train' in v]
        infer_loader_list = [
            v for v in split_info.keys() if not ('train' in v)]

        cfg_module = importlib.import_module('models.hovernet.opt')
        cfg_getter = getattr(cfg_module, 'get_config')

        with open(f'{save_path}/settings.yml', 'w') as fptr:
            yaml.dump(template_paramset, fptr, default_flow_style=False)

        model_config = cfg_getter(
                            train_loader_list,
                            infer_loader_list,
                            **template_paramset)

        def create_dataset(
                run_mode=None, subset_name=None, setup_augmentor=None):
            target_gen_func = getattr(
                importlib.import_module('models.hovernet.targets'),
                'gen_targets'
            )
            img_path = f'{WORKSPACE_DIR}/data/images.npy'
            ann_path = f'{WORKSPACE_DIR}/data/labels.npy'
            indices = split_info[subset_name]
            return FileLoader(
                        img_path,
                        ann_path,
                        indices,
                        with_type=True,
                        input_shape=[256, 256],
                        mask_shape=[256, 256],
                        run_mode=run_mode,
                        target_gen_func=[target_gen_func, {}]
                    )

        run_kwargs = {
            'seed': seed,
            'debug': False,
            'logging': True,
            'log_dir': save_path + '/model/',
            'create_dataset': create_dataset,
            'model_config': model_config,
        }

        from run_train import RunManager
        trainer = RunManager(**run_kwargs)
        trainer.run()

    save_path_ = f'{SAVE_ROOT}/{FOLD_IDX:02d}/'
    split_info = splits[FOLD_IDX]
    run_one_split_with_param_set(save_path_, split_info, {})
