# make sure you're logged in with \`huggingface-cli login\`
import torch
import os
from torchvision.utils import make_grid, save_image
from PIL import Image
import os
import hydra
from hydra.core.hydra_config import HydraConfig
import lpips
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import transforms
import time

from models.stable_diffusion.diffusers import StableDiffusionParticleInversePipeline, DDIMScheduler

from utils.degredations import build_degredation_model, get_degreadation_image
from utils.functions import postprocess, preprocess
from datasets import build_loader
from utils.torch_utils import get_logger, init_omega, seed_everything
from algos import build_algo
from utils.save import save_result, save_imagenet_result_particles


# TODO correct this
@hydra.main(version_base="1.2", config_path="_configs", config_name="ddrmpp_ffhq_stable")
def main(cfg: DictConfig):
    ####
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    # Set cuda device
    torch.cuda.set_device(cfg.exp.gpu)
    
    # Generator for stable diffusion
    seed_everything(cfg.exp.seed)
    generator = torch.Generator(device="cuda").manual_seed(cfg.exp.seed)

    # Load config file
    cwd = HydraConfig.get().runtime.output_dir
    cfg = init_omega(cfg, cwd)

    # Set folder name
    cfg.algo.name_folder = cfg.algo.name + str(cfg.algo.gamma)

    # Build paths
    deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, cfg.algo.name_folder)
    if not os.path.exists(deg_path):
        os.makedirs(deg_path)
    
    output_path = os.path.join(cfg.exp.root, cfg.exp.output_path, cfg.algo.name_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    evol_path = os.path.join(cfg.exp.root, cfg.exp.evol_path, cfg.algo.name_folder)
    if not os.path.exists(evol_path):
        os.makedirs(evol_path)

    dataset_name = cfg.dataset.name

    # Get logger
    logger = get_logger(name="main", cfg=cfg)
    logger.info(f'Experiment name is {"RSD"}')

    # Load model
    pipe = StableDiffusionParticleInversePipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    # pipe = StableDiffusionParticleInversePipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(cfg.exp.num_steps, device=cfg.exp.gpu)
    ts = pipe.scheduler.timesteps
    logger.info(f'Model loaded')

    if cfg.algo.dino_flag is True:
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to("cuda")
        lpips_ = lpips.LPIPS(net='alex').cuda() # best forward score
    else:
        dino = None
        lpips_ = lpips.LPIPS(net='alex').cuda() # best forward score

    # Load degradation model
    algo = build_algo(pipe, cfg)
    H = algo.H
    logger.info(f'Degradation model loaded. The experiment corresponds to {cfg.algo.deg}')

    ### UP TO THIS POINT, EVERYTHING IS THE SAME EITHER FOR STABLE DIFFUSION OR OTHER DIFFUSION MODELS.

    if cfg.algo.name == 'rsd_stable' or cfg.algo.name == 'rsd_stable_nonaug':
        #
        # If we have an image as input, we have a single run. We do it separately.
        #
        if cfg.exp.load_img_id is True:
            img_path = os.path.join(cfg.exp.img_path, cfg.exp.img_id)
            img = Image.open(img_path).convert('RGB')
            transform = transforms.Compose([ 
                transforms.ToTensor() 
            ]) 
            x = transform(img) 
            x = x.cuda()

            # If 256 size, upsample to 512
            if x.shape[-1] == 256:
                upsample = nn.Upsample(scale_factor=2, mode='nearest') 
                x = upsample(x.unsqueeze(0)).squeeze()
            elif x.shape[-1] != 512 or x.shape[-2] != 512:
                x = transforms.Resize((512, 512))(x)
                print(x.shape)

            if cfg.algo.n_particles > 1:
                x = x.repeat(cfg.algo.n_particles, 1, 1, 1)
            else:
                x = x.unsqueeze(0)

            # Get noisy measurement
            y = None
            x = preprocess(x)
            y_0 = H.H(x.to("cuda"))
            y_0 = y_0 + torch.randn_like(y_0) * cfg.algo.sigma_x0    #?? what is it for???
            logger.info(f'Images loaded and measurements generated')

            # If save deg true, then we save the degradated image
            if cfg.exp.save_deg is True:
                if cfg.algo.deg == 'deblur_motion':
                    xo = postprocess(y_0)
                else:
                    xo = postprocess(get_degreadation_image(y_0, H, cfg))
                transform_pil_to_image = transforms.ToPILImage()
                img = transform_pil_to_image(xo[0].cpu())
                deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, 'x_deg.png')
                save_image(xo[0].cpu(), deg_path)
                logger.info(f'Degradated image y saved in {deg_path}')
                

            # Run the algorithm
            logger.info(f'Running setting with coeff: {cfg.algo.gamma}, \
                        lr: {cfg.algo.lr_x, cfg.algo.lr_z}, \
                        rho_reg: {cfg.algo.rho_reg}, \
                        w_t: {cfg.algo.w_t}, \
                        sigma_break: {cfg.algo.sigma_break}')
            
            start = time.time()
            
            if cfg.algo.dino_flag is True:
                xt_s, _ = algo.sample(x, y, ts, generator, y_0, dino = dino)
            else:
                xt_s, _ = algo.sample(x, y, ts, generator, y_0)  
            
            end = time.time()
            print(f"Time taken: {end - start}")

            ## !! This is for saving at 256 resolution
            
            # if isinstance(xt_s, list):
            #     xo = postprocess(xt_s[0]).cpu()
            #     downsample = nn.AvgPool2d(2, 2)
            #     for i in range(cfg.algo.n_particles):
            #         xo[0,i,:,:,:] = downsample(xo[i,:,:,:].unsqueeze(0)).squeeze()
            #     x_gt = postprocess(x_256)
            # else:
            #     xo = postprocess(xt_s).cpu()
            #     downsample = nn.AvgPool2d(2, 2)
            #     aux = torch.zeros((cfg.algo.n_particles, 3, 256, 256))
            #     for i in range(cfg.algo.n_particles):
            #         aux[i,:,:,:] = downsample(xo[i,:,:,:].unsqueeze(0)).squeeze()
            #     xo = aux
            #     x_gt = postprocess(x_256)

            if isinstance(xt_s, list):
                xo = postprocess(xt_s[0]).cpu()
                x_gt = postprocess(x)
            else:
                xo = postprocess(xt_s).cpu()
                x_gt = postprocess(x)

            # Compute PSNR
            mse = torch.mean((xo - x_gt.cpu()) ** 2, dim=(1, 2, 3))
            psnr = 10 * torch.log10(1 / (mse + 1e-10))
            logger.info(f'Mean PSNR: {psnr.mean()}')
            logger.info(f'PSNR: {psnr}')
            
            # Compute LPIPS
            LPIPS = lpips_(xo.cuda(), x_gt.cuda())
            logger.info(f'Mean LPIPS: {LPIPS.mean()}')
            logger.info(f'LPIPS: {LPIPS[:,0,0,0]}')

            output_path_img = f'{output_path}/{cfg.exp.img_id.split('.')[0]}'
            if not os.path.exists(output_path_img):
                os.makedirs(output_path_img)

            for i in range(cfg.algo.n_particles):
                save_image(xo[i], f'{output_path_img}/x_hat_{i}.png')
            image_grid = make_grid(xo.cpu())
            save_image(image_grid, f'{output_path_img}/x_hat_grid.png')

            torch.cuda.empty_cache()
            logger.info(f'Done. You can fine the generated images in {output_path_img}')

        else:
            # We iterate over the data loader
            loader = build_loader(cfg)
            logger.info(f'Dataset size is {len(loader.dataset)}')

            output_path = os.path.join(output_path, cfg.algo.deg)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for it, (x, y, info) in enumerate(loader):
                # if info['index'][0].split('.')[0] < '00035':
                #     continue
                logger.info(f"Input image:{ info['index'][0]}")
                
                # If 256 size, upsample to 512
                if x.shape[-1] == 256:
                    upsample = nn.Upsample(scale_factor=2, mode='nearest') 
                    x = upsample(x).squeeze()
                elif x.shape[-1] != 512 or x.shape[-2] != 512:
                    x = transforms.Resize((512, 512))(x)
                    # print(x.shape)

                x = x.cuda()
                y = y.cuda()
                x = preprocess(x)
                kwargs = info

                if cfg.algo.n_particles > 1:
                    x = x.repeat(cfg.algo.n_particles, 1, 1, 1)
                else:
                    x = x.unsqueeze(0)
                
                print(x.shape)
                # Get noisy measurement
                y_0 = H.H(x.to("cuda"))
                y_0 = y_0 + torch.randn_like(y_0) * cfg.algo.sigma_x0    #?? what is it for???
                logger.info(f'Images loaded and measurements generated')

                # If save deg true, then we save the degradated image
                if cfg.exp.save_deg is True:
                    if cfg.algo.deg == 'deblur_motion':
                        xo = postprocess(y_0)
                    else:
                        xo = postprocess(get_degreadation_image(y_0, H, cfg))
                    transform_pil_to_image = transforms.ToPILImage()
                    img = transform_pil_to_image(xo[0].cpu())
                    deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, 'x_deg.png')
                    save_image(xo[0].cpu(), deg_path)
                    logger.info(f'Degradated image y saved in {deg_path}')
                    

                # Run the algorithm
                logger.info(f'Running setting with coeff: {cfg.algo.gamma}, \
                            lr: {cfg.algo.lr_x, cfg.algo.lr_z}, \
                            rho_reg: {cfg.algo.rho_reg}, \
                            w_t: {cfg.algo.w_t}, \
                            sigma_break: {cfg.algo.sigma_break}')
                
                
                if cfg.algo.dino_flag is True:
                    xt_s, _ = algo.sample(x, y, ts, generator, y_0, dino = dino)
                else:
                    xt_s, _ = algo.sample(x, y, ts, generator, y_0)  
                
                if isinstance(xt_s, list):
                    xo = postprocess(xt_s[0]).cpu()
                    x_gt = postprocess(x)
                else:
                    xo = postprocess(xt_s).cpu()
                    x_gt = postprocess(x)

                save_result(dataset_name, xo, y, info, output_path, "")

                # Compute PSNR
                mse = torch.mean((xo - x_gt.cpu()) ** 2, dim=(1, 2, 3))
                psnr = 10 * torch.log10(1 / (mse + 1e-10))
                print(f'Mean PSNR: {psnr.mean()}')
                print(f'PSNR: {psnr}')
                
                # Compute LPIPS
                LPIPS = lpips_(xo.cuda(), postprocess(x))
                print(f'Mean LPIPS: {LPIPS.mean()}')
                print(f'LPIPS: {LPIPS[:,0,0,0]}')

                torch.cuda.empty_cache()
                logger.info(f'Done. You can fine the generated images in {output_path}')
    else:
        raise NotImplementedError('Wrong method')


if __name__ == "__main__":
    main()
