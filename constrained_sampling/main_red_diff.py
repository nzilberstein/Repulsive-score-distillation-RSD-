# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import datetime
import logging
import os
import shutil
import sys
import time
from pathlib import Path


import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import distutils.version
import torch.utils.tensorboard as tb
import torchvision.utils as tvu
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.utils import make_grid, save_image
# import wandb
# import pickle
# import pandas as pd
import torchvision.transforms as transforms

import lpips

from algos import build_algo
from datasets import build_loader
from models import build_model
from models.classifier_guidance_model import ClassifierGuidanceModel
from models.diffusion import Diffusion
from utils.distributed import get_logger, init_processes, common_init
from utils.functions import get_timesteps, postprocess, preprocess, strfdt
from utils.degredations import get_degreadation_image
from utils.torch_utils import init_omega, seed_everything
from utils.save import save_result, save_imagenet_result_particles



@hydra.main(version_base="1.2", config_path="_configs", config_name="ddrmpp_ffhq")
def main(cfg: DictConfig):
    ####
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"

    torch.cuda.set_device(cfg.exp.gpu)

    # Generator for stable diffusion
    seed_everything(cfg.exp.seed)

    # Load config file
    cwd = HydraConfig.get().runtime.output_dir
    cfg = init_omega(cfg, cwd)

    # Build paths
    deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, cfg.algo.name)
    if not os.path.exists(deg_path):
        os.makedirs(deg_path)
    
    output_path = os.path.join(cfg.exp.root, cfg.exp.output_path, cfg.algo.name, cfg.algo.deg)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    evol_path = os.path.join(cfg.exp.root, cfg.exp.evol_path, cfg.algo.name)
    if not os.path.exists(evol_path):
        os.makedirs(evol_path)

    dataset_name = cfg.dataset.name

    # Get logger
    logger = get_logger(name="main", cfg=cfg)
    logger.info(f'Experiment name is {"RED-diff"}')
    
    model, classifier = build_model(cfg)
    model.eval()
    if classifier is not None:
        classifier.eval()

    diffusion = Diffusion(**cfg.diffusion)
    cg_model = ClassifierGuidanceModel(model, classifier, diffusion, cfg)   #?? what is the easiest way to call stable diffusion?
    ts = get_timesteps(cfg)

    # Load dino and lpips: TODO: do in another part.
    if cfg.algo.dino_flag is True:
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to("cuda")
        lpips_ = lpips.LPIPS(net='alex').cuda() # best forward score
    else:
        dino = None
        lpips_ = lpips.LPIPS(net='alex').cuda() # best forward score

    algo = build_algo(cg_model, cfg)
    if "ddrm" in cfg.algo.name or "mcg" in cfg.algo.name or "dps" in cfg.algo.name or "pgdm" in cfg.algo.name or "reddiff" in cfg.algo.name or "reddiff_svgd" in cfg.algo.name or "reddiff_svgd_particles" in cfg.algo.name:
        H = algo.H

    if cfg.exp.load_img_id is True:
        img_path = os.path.join(cfg.exp.img_path, cfg.exp.img_id)
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([ 
            transforms.ToTensor() 
        ]) 
        x = transform(img) 
        x = x.cuda()

        x = preprocess(x)

        if cfg.algo.n_particles > 1:
                x = x.repeat(cfg.algo.n_particles, 1, 1, 1)
            
        if "ddrm" in cfg.algo.name or "mcg" in cfg.algo.name or "dps" in cfg.algo.name or "pgdm" in cfg.algo.name or "reddiff" in cfg.algo.name:
            y_0 = H.H(x)
            y_0 = y_0 + torch.randn_like(y_0) * cfg.algo.sigma_y  #?? what is it for???
            kwargs = {'y_0': y_0}
            y = None

        # If save deg true, then we save the degradated image
        if cfg.exp.save_deg is True:
            xo = postprocess(get_degreadation_image(y_0, H, cfg))
            transform_pil_to_image = transforms.ToPILImage()
            img = transform_pil_to_image(xo[0].cpu())
            deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, 'x_deg.png')
            save_image(xo[0].cpu(), deg_path)
            logger.info(f'Degradated image y saved in {deg_path}')

        xt_s, _ = algo.sample(x, y, ts, **kwargs)                  
        
        if isinstance(xt_s, list):
            xo = postprocess(xt_s[0]).cpu()
            x_gt = postprocess(x)
        else:
            xo = postprocess(xt_s).cpu()
            x_gt = postprocess(x)

        # PSNR
        mse = torch.mean((xo - x_gt.cpu()) ** 2, dim=(1, 2, 3))
        psnr = 10 * torch.log10(1 / (mse + 1e-10))
        print(f'Mean PSNR: {psnr.mean()}')
        print(f'PSNR: {psnr}')
        
        # LPIPS
        LPIPS = lpips_(xo.cuda(), postprocess(x))
        print(f'Mean LPIPS: {LPIPS.mean()}')
        print(f'LPIPS: {LPIPS[:,0,0,0]}')

        output_path_img = f'{output_path}/{cfg.exp.img_id.split('.')[0]}'
        if not os.path.exists(output_path_img):
            os.makedirs(output_path_img)

        for i in range(cfg.algo.n_particles):
            save_image(xo[i], f'{output_path_img}/x_hat_{i}.png')
        image_grid = make_grid(xo.cpu())
        save_image(image_grid, f'{output_path_img}/x_hat_grid.png')

        # raise NotImplementedError("Loading image id is not implemented yet.")
    else:
        loader = build_loader(cfg)
        logger.info(f'Dataset size is {len(loader.dataset)}')
    
        start_time = time.time()
        for it, (x, y, info) in enumerate(loader):
            logger.info(f"Input image:{ info['index'][0]}")
            x = x.cuda()
            y = y.cuda()
            x = preprocess(x)
            kwargs = info

            
            if cfg.algo.n_particles > 1:
                x = x.repeat(cfg.algo.n_particles, 1, 1, 1)
            
            if "ddrm" in cfg.algo.name or "mcg" in cfg.algo.name or "dps" in cfg.algo.name or "pgdm" in cfg.algo.name or "reddiff" in cfg.algo.name:
                y_0 = H.H(x)
                y_0 = y_0 + torch.randn_like(y_0) * cfg.algo.sigma_y  #?? what is it for???
                kwargs["y_0"] = y_0

            # print(kwargs)
            # If save deg true, then we save the degradated image
            if cfg.exp.save_deg is True:
                xo = postprocess(get_degreadation_image(y_0, H, cfg))
                transform_pil_to_image = transforms.ToPILImage()
                img = transform_pil_to_image(xo[0].cpu())
                deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, 'x_deg.png')
                save_image(xo[0].cpu(), deg_path)
                logger.info(f'Degradated image y saved in {deg_path}')
 
            xt_s, _ = algo.sample(x, y, ts, **kwargs)                  
            
            if isinstance(xt_s, list):
                xo = postprocess(xt_s[0]).cpu()
                x_gt = postprocess(x)
            else:
                xo = postprocess(xt_s).cpu()
                x_gt = postprocess(x)

            save_result(dataset_name, xo, y, info, output_path, "")

            # PSNR
            mse = torch.mean((xo - x_gt.cpu()) ** 2, dim=(1, 2, 3))
            psnr = 10 * torch.log10(1 / (mse + 1e-10))
            print(f'Mean PSNR: {psnr.mean()}')
            print(f'PSNR: {psnr}')
            
            # LPIPS
            LPIPS = lpips_(xo.cuda(), postprocess(x))
            print(f'Mean LPIPS: {LPIPS.mean()}')
            print(f'LPIPS: {LPIPS[:,0,0,0]}')
            
    # if len(loader) > 0:
    #     psnrs = torch.cat(psnrs, dim=0)
    #     logger.info(f'Mean PSNR: {psnrs.mean().item()}')
        
    # logger.info("Done.")
    # now = time.time() - start_time
    # now_in_hours = strfdt(datetime.timedelta(seconds=now))
    # logger.info(f"Total time: {now_in_hours}")

if __name__ == "__main__":
    main()
