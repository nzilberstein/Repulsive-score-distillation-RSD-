import logging
import os
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig


def init_omega(cfg, cwd):
    """ Initialize the distributed environment. """
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.cwd = cwd

    return cfg

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_logger(name=None, cfg=None):
    load_path = os.path.join(cfg.cwd, ".hydra/hydra.yaml")
    hydra_conf = OmegaConf.load(load_path)
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))

    return logging.getLogger(name)


def get_results_file(cfg, logger):
    if dist.get_rank() == 0 or not dist.is_available():
        results_root = os.path.join(cfg.exp.root, 'results')
        os.makedirs(results_root, exist_ok=True)
        if '/' in cfg.results:
            results_dir = '/'.join(cfg.results.split('/')[:-1])
            results_dir = os.path.join(results_root, results_dir)
            logger.info(f'Creating directory {results_dir}')
            os.makedirs(results_dir, exist_ok=True)
        results_file = f'{results_root}/{cfg.results}.yaml'
    return results_file
