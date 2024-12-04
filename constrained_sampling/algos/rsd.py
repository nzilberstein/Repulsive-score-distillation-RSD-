# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import os 
import shutil
from torchvision.utils import save_image, make_grid

from models.stable_diffusion.diffusers import StableDiffusionParticleInversePipeline
from utils.degredations import build_degredation_model
from utils.functions import postprocess
from .ddim import DDIM

import matplotlib.pyplot as plt
import numpy as np

import lpips
import torch.nn as nn
from tqdm import tqdm, trange
import random

class RSD(DDIM):
    def __init__(self, model: StableDiffusionParticleInversePipeline, cfg: DictConfig):
        self.model = model
        # self.diffusion = model.diffusion
        self.H = build_degredation_model(cfg)
        self.cfg = cfg
        # self.awd = cfg.algo.awd
        # self.cond_awd = cfg.algo.cond_awd
        self.w_t = cfg.algo.w_t
        self.obs_weight = cfg.algo.obs_weight
        self.lr_x = cfg.algo.lr_x
        self.lr_z = cfg.algo.lr_z
        self.denoise_term_weight = cfg.algo.denoise_term_weight
        self.sigma_x0 = cfg.algo.sigma_x0
        self.rho_reg = cfg.algo.rho_reg
        self.dino_flag = cfg.algo.dino_flag
        self.gamma = cfg.algo.gamma
        self.load_img = cfg.exp.load_img_id
        self.sigma_break = cfg.algo.sigma_break

        self.height = 512
        self.width = 512
        self.num_channels_latents = cfg.model.C
        self.device = self.model._execution_device
        self.all_sigmas = torch.sqrt((1 - self.model.scheduler.alphas_cumprod) / self.model.scheduler.alphas_cumprod)

        self.prompt_embeds = self.model._encode_prompt([''] * cfg.algo.n_particles, self.device, num_images_per_prompt = 1, do_classifier_free_guidance=False)

        self.lpips = lpips.LPIPS(net='alex').cuda() # best forward scores

    def sample(self, x, y, ts, generator, y_0, dino = None, results256 = False):
        if len(x.shape) == 3:
            batch_size = 1
        else:
            batch_size = x.size(0)
        H = self.H
        input_img = x

        #optimizer
        x_init = torch.randn((batch_size, 3, self.height, self.width)).cuda()
        # print(y_0.shape)
        # if self.cfg.algo.deg == "phase_retrieval":
        #     x_init = self.initialize(x, y, ts, y_0=y_0).cuda()
        # x_init = y_0.clone().cuda()
        x = torch.autograd.Variable(x_init, requires_grad=True)   #, device=device).type(dtype)
        optimizer_x = torch.optim.Adam([x], lr=self.lr_x, betas=(0.9, 0.99), weight_decay=0.0)   #original: 0.999

        latents = self.model.prepare_latents(batch_size, self.num_channels_latents, self.height, self.width, dtype=self.prompt_embeds.dtype, device=self.device, generator=generator)
        latents = torch.autograd.Variable(latents, requires_grad=True).cuda() 
        optimizer_z = torch.optim.Adam([latents], lr=self.lr_z, betas=(0.9, 0.99), weight_decay=0.0)   #original: 0.999

        counter = 0
        if self.load_img:
            evol_path = f'/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/_exp/evol/rsd_stable_single_{self.cfg.algo.deg}_{self.cfg.exp.img_id}_{self.cfg.algo.gamma}_{self.lr_z}'
        else:
            evol_path = '/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/_exp/evol/rsd_stable'
    
        if not os.path.exists(f'{evol_path}'):
            os.makedirs(f'{evol_path}')
        else:
            shutil.rmtree(f'{evol_path}')
            os.makedirs(f'{evol_path}')
        
        x_list = []


        sigmas = torch.sqrt((1 - self.model.scheduler.alphas_cumprod) / self.model.scheduler.alphas_cumprod)
        # Flip the order of sigmas
        # sigmas = sigmas.flip(0)
        sigmas = sigmas[self.model.scheduler.timesteps.cpu().numpy()]
        
        # print(sigmas.flip(0))
        # print(sigmas[self.model.scheduler.timesteps.cpu().numpy()])

        idx_sigma_break = int(((self.model.scheduler.timesteps.shape[0])/ 1000 ) * self.sigma_break)

        for i, t in tqdm(enumerate(ts[:-1])):

            # tensor_list = ts.tolist()
            # t_np = random.choice(tensor_list)
            # i = tensor_list.index(t_np)
            # t = torch.tensor(t_np)


            noise_t = torch.randn_like(latents).cuda()
            latent_pred_t = self.model.scheduler.add_noise(latents, noise_t, t)
            latent_pred_t = self.model.scheduler.scale_model_input(latent_pred_t, t)


            alpha_t = self.model.scheduler.alphas_cumprod[self.model.scheduler.timesteps[i].cpu().numpy()]
            alpha_t.requires_grad_(True)

            with torch.no_grad():
                et = self.model.unet(latent_pred_t, t, encoder_hidden_states=self.prompt_embeds, cross_attention_kwargs=None).sample
                        
            et = et.detach()
            noise_pred = et - noise_t

            
            # Repulsion term
            if self.dino_flag is True and sigmas[i] > sigmas[idx_sigma_break]:
            # if self.dino_flag is True and sigmas[i] < sigmas[idx_sigma_break]:
                if counter % 50 == 0:
                    print(f'Using DINO at step {counter}')
                latent_pred_t.requires_grad_(True)
                dino.requires_grad_(True)
                self.model.vae.decoder.requires_grad_(True)
                dino.train()
                self.model.vae.train()

                # x_pred_z = self.model.decode_latents(latents, stay_on_device=True)
                x_pred_z = self.model.decode_latents(latent_pred_t, stay_on_device=True)

                dino_out = dino(x_pred_z)

                latents_vec = dino_out.view(len(dino_out), -1)
                diff = latents_vec.unsqueeze(1) - latents_vec.unsqueeze(0)
                diff = diff[~torch.eye(diff.shape[0], dtype=bool)].view(diff.shape[0], -1, diff.shape[-1])

                distance = torch.norm(diff, p=2, dim=-1, keepdim=True)
                num_images = latents_vec.shape[0]
                h_t = (distance.median(dim=1, keepdim=True)[0]) ** 2 / np.log(
                    num_images - 1)
                weights = torch.exp(- (distance ** 2 / h_t))

                grad_phi = 2 * weights * diff / h_t
                grad_phi = grad_phi.sum(dim=1)

                eval_sum = torch.sum(dino_out * grad_phi.detach())
                # deps_dx_backprop = torch.autograd.grad(eval_sum, latents)[0]
                deps_dx_backprop = torch.autograd.grad(eval_sum, latent_pred_t)[0]
                grad_phi = deps_dx_backprop.view_as(latents)

                K_svgd_z_mat_reg_sum = weights.sum(dim = 1)
                nabla_log = torch.div(grad_phi, K_svgd_z_mat_reg_sum.unsqueeze(-1).unsqueeze(-1))

                # nabla_log = nabla_log[0:int(batch_size/2),:,:,:] # For hybrid mode
                noise_pred = et - noise_t - self.gamma * (1-alpha_t).sqrt() * nabla_log
                # noise_pred = et - noise_t - self.gamma * sigmas[i] * nabla_log

            else:
                noise_pred = et - noise_t

            loss_diffusion = torch.mul((noise_pred).detach(), latents).mean()
                        
            # Weighting
            # snr_inv = (1-alpha_t)
            snr_inv = (1-alpha_t).sqrt() / alpha_t.sqrt()
            rho_reg = self.rho_reg # * (1-alpha_t).detach()

            
            ## Optimize z ##
            # encoded_z_0 = self.model.vae.encode(x).latent_dist.sample(generator)
            x_pred_z = self.model.decode_latents(latents, stay_on_device=True)
            e_decod = x - x_pred_z
            loss_reg = (e_decod**2).mean() / 2
            # e_encod = encoded_z_0 - latents
            # loss_encod = (e_encod**2).mean() / 2

            loss_z = loss_reg + (self.w_t) * snr_inv * loss_diffusion

            optimizer_z.zero_grad()
            loss_z.backward()
            optimizer_z.step()

            ## Optimize x ##
            x_pred_z = self.model.decode_latents(latents, stay_on_device=True).detach()
            e_decod = x - x_pred_z
            loss_reg = (e_decod**2).mean() / 2

            e_obs = y_0 - H.H(x)
            loss_obs = (e_obs**2).mean() / 2

            loss_x = loss_obs + (rho_reg) *  loss_reg

            optimizer_x.zero_grad()
            loss_x.backward()
            optimizer_x.step()


            # #save for visualization
            if self.cfg.exp.save_evolution:
                if counter % 50 == 0:
                    # print(rho_reg)
                    image_evol = make_grid((postprocess(x).clone().detach().cpu()))
                    save_image(image_evol, f'{evol_path}/evol_{counter}.png')      
    
                    image_z = self.model.decode_latents(latents, stay_on_device=True).detach()
                    image_evol = make_grid((postprocess(image_z).clone().detach().cpu()))
                    save_image(image_evol, f'{evol_path}/evol_{counter}_z.png')     

                    if results256 == True:
                        downsample = nn.AvgPool2d(2, 2)
                        aux = torch.zeros((self.cfg.algo.n_particles, 3, 256, 256))
                        x_256 = torch.zeros((self.cfg.algo.n_particles, 3, 256, 256))
                        xx = postprocess(x).clone().detach().cpu()
                        for q in range(self.cfg.algo.n_particles):
                            aux[q,:,:,:] = downsample(xx[q, :,:,:].unsqueeze(0)).squeeze()
                            x_256[q,:,:,:] = downsample(postprocess(input_img[q,:,:,:].unsqueeze(0))).squeeze()
                        xo = aux
                    else:
                        # print(len(x_list))
                        xo = postprocess(x).clone().detach().cpu()
                        x_256 = postprocess(input_img).cpu()

                    x_list.append(x.clone().detach())

                    # PSNR
                    mse = torch.mean((xo - x_256.cpu()) ** 2, dim=(1, 2, 3))
                    psnr = 10 * torch.log10(1 / (mse + 1e-10))
                    print(f'Mean PSNR: {psnr.mean()}')
                    print(f'PSNR: {psnr}')
                    
                    # LPIPS
                    LPIPS = self.lpips(xo.cuda(), x_256.cuda())
                    print(f'Mean LPIPS: {LPIPS.mean()}')
                    print(f'LPIPS: {LPIPS[:,0,0,0]}')
                
                if counter > len(ts) - 100 - 1 and counter % 10 == 0:
                    image_evol = make_grid((postprocess(x).clone().detach().cpu()))
                    save_image(image_evol, f'{evol_path}/evol_{counter}.png')      

                    image_z = self.model.decode_latents(latents, stay_on_device=True).detach()
                    image_evol = make_grid((postprocess(image_z).clone().detach().cpu()))
                    save_image(image_evol, f'{evol_path}/evol_{counter}_z.png') 
                                    
                    if results256 == True:
                        downsample = nn.AvgPool2d(2, 2)
                        aux = torch.zeros((self.cfg.algo.n_particles, 3, 256, 256))
                        x_256 = torch.zeros((self.cfg.algo.n_particles, 3, 256, 256))
                        xx = postprocess(x).clone().detach().cpu()
                        for q in range(self.cfg.algo.n_particles):
                            aux[q,:,:,:] = downsample(xx[q, :,:,:].unsqueeze(0)).squeeze()
                            x_256[q,:,:,:] = downsample(postprocess(input_img[q,:,:,:].unsqueeze(0))).squeeze()
                        xo = aux
                    else:
                        
                        xo = postprocess(x).clone().detach().cpu()
                        x_256 = postprocess(input_img).cpu()

                    # x_256 = downsample(input_img[0,:,:,:].unsqueeze(0)).squeeze()

                    # PSNR
                    mse = torch.mean((xo - x_256.cpu()) ** 2, dim=(1, 2, 3))
                    psnr = 10 * torch.log10(1 / (mse + 1e-10))
                    print(f'Mean PSNR: {psnr.mean()}')
                    print(f'PSNR: {psnr}')
                    
                    # LPIPS
                    LPIPS = self.lpips(xo.cuda(), x_256.cuda())
                    print(f'Mean LPIPS: {LPIPS.mean()}')
                    print(f'LPIPS: {LPIPS[:,0,0,0]}')

                    x_list.append(x.clone().detach())
            else:
                x_list.append(x.clone().detach())
            counter = counter + 1
                
        x_list.append(x.clone().detach())
        return x_list[-1], x 
        
    def initialize(self, x, y, ts, **kwargs):
        y_0 = kwargs['y_0']
        H = self.H
        x_0 = H.H_pinv(y_0).view(*x.size()).detach()
        return x_0   #alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)    #x_0


    def plot_weight_den(self, ts, **kwargs):
    
        #ts.reverse()
        alpha = self.diffusion.alpha(torch.tensor(ts).cuda())
        
        snr_inv = (1-alpha).sqrt()/alpha.sqrt()  #1d torch tensor
        snr_inv = snr_inv.detach().cpu().numpy()
            
        # plot lines
        plt.plot(ts, snr_inv, label = "1/snr", linewidth=2)
        plt.plot(ts, np.sqrt(snr_inv), label = "sqrt(1/snr)", linewidth=2)
        #plt.plot(ts, np.power(snr_inv, 2/3), label = "(1/snr)^2/3")
        plt.plot(ts, np.square(snr_inv), label = "square(1/snr)", linewidth=2)
        plt.plot(ts, np.log(snr_inv+1), label = "log(1+1/snr)", linewidth=2)   #ln
        plt.plot(ts, np.clip(snr_inv, None, 1), label = "clip(1/snr,max=1)", linewidth=2)
        plt.plot(ts, np.power(snr_inv, 0.0), label = "const", linewidth=2)

        plt.legend()
        #plt.xscale('log')
        plt.yscale('log')
        plt.xlim(max(ts), min(ts))
        plt.xlabel("timestep", fontsize = 15)
        plt.ylabel("denoiser weight", fontsize = 15)
        
        plt.legend(fontsize = 13)
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)

        plt.savefig('weight_type_vs_step.png')

        return 0







