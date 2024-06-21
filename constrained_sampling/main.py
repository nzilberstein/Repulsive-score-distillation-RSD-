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

from models.stable_diffusion.diffusers import StableDiffusionParticleInversePipeline, DDIMScheduler

from utils.degredations import build_degredation_model, get_degreadation_image
from utils.functions import postprocess, preprocess
from datasets import build_loader
from utils.torch_utils import get_logger, init_omega, seed_everything


# TODO correct this
@hydra.main(version_base="1.2", config_path="_configs", config_name="ddrmpp_ffhq_stable")
def main(cfg: DictConfig):
    ####
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"

    torch.cuda.set_device(cfg.exp.gpu)

    # Generator for stable diffusion
    seed_everything(cfg.exp.seed)
    generator = torch.Generator(device="cuda").manual_seed(cfg.exp.seed)

    # Load config file
    cwd = HydraConfig.get().runtime.output_dir
    cfg = init_omega(cfg, cwd)

    # Build paths
    deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, 'x_deg.png')
    
    output_path = os.path.join(cfg.exp.root, cfg.exp.output_path, cfg.algo.name, cfg.algo.deg)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    evol_path = os.path.join(cfg.exp.root, cfg.exp.evol_path)

    # Get logger
    logger = get_logger(name="main", cfg=cfg)
    logger.info(f'Experiment name is {"RSD"}')

    # Load model
    pipe = StableDiffusionParticleInversePipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    logger.info(f'Model loaded')

    # Load dino and lpips: TODO: do in another part.
    if cfg.algo.dino_flag is True:
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to("cuda")
        lpips_ = lpips.LPIPS(net='alex').cuda() # best forward score
    else:
        dino = None
        lpips_ = lpips.LPIPS(net='alex').cuda() # best forward score

    # Load degradation model
    H = build_degredation_model(cfg)
    logger.info(f'Degradation model loaded. The experiment corresponds to {cfg.algo.deg}')

    ### UP TO THIS POINT, EVERYTHING IS THE SAME EITHER FOR STABLE DIFFUSION OR OTHER DIFFUSION MODELS.

    if cfg.algo.name == 'rsd_stable':
        # If we have an image as input, we have a single run. We do it separately.
        if cfg.exp.load_img_id is True:
            # Load image from file
            img_path = os.path.join(cfg.exp.img_path, cfg.exp.img_id)
            img = Image.open(img_path).convert('RGB')

            # Convert to tensor
            transform = transforms.Compose([ 
                transforms.ToTensor() 
            ]) 
            x = transform(img) 

            # If 256 size, upsample to 512
            if x.shape[-1] == 256:
                upsample = nn.Upsample(scale_factor=2, mode='nearest') 
                x = upsample(x.unsqueeze(0)).squeeze()

            # Get noisy measurement. TODO: Do a separate function
            x = preprocess(x)
            y_0 = H.H(x.unsqueeze(0).to("cuda"))
            y_0 = y_0 + torch.randn_like(y_0) * cfg.algo.sigma_x0    #?? what is it for???
            logger.info(f'Image loaded')

            # If save deg true, then we save the degradated image
            if cfg.exp.save_deg is True:
                xo = postprocess(get_degreadation_image(y_0, H, cfg))
                transform_pil_to_image = transforms.ToPILImage()
                img = transform_pil_to_image(xo[0].cpu())
                deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, 'x_deg.png')
                save_image(xo[0].cpu(), deg_path)
                logger.info(f'Degradated image y saved in {deg_path}')
                
            # Prompt empty for stable diffusion
            prompt = ['']
            num_of_image = cfg.algo.n_particles
            prompt = prompt * num_of_image

            # Run the algorithm
            logger.info(f'Running setting with coeff: {cfg.algo.gamma}, \
                        lr: {cfg.algo.lr_x, cfg.algo.lr_z}, \
                        rho_reg: {cfg.algo.rho_reg}, \
                        w_t: {cfg.algo.w_t}')
            
            out = pipe.sample(prompt, 
                            generator=generator, 
                            num_inference_steps=cfg.exp.num_steps, 
                            gamma=cfg.algo.gamma, 
                            guidance_scale=cfg.algo.w, 
                            dino=dino, 
                            lr_x = cfg.algo.lr_x, 
                            lr_z = cfg.algo.lr_z, 
                            output_type='tensor',
                            H = H, 
                            y_0 = y_0, 
                            rho_reg=cfg.algo.rho_reg, 
                            w_t=cfg.algo.w_t, 
                            evol_path = evol_path)

            images_DINO = postprocess(out.images)
            list_images = [images_DINO.cpu()]
            image_grid = make_grid(images_DINO.cpu())

            # Compute metrics. TODO: separate function
            # PSNR
            mse = torch.mean((images_DINO - postprocess(x).cuda().unsqueeze(0).repeat(num_of_image, 1, 1, 1)) ** 2, dim=(1, 2, 3))
            psnr = 10 * torch.log10(1 / (mse + 1e-10))

            # LPIPS
            LPIPS = lpips_(images_DINO, postprocess(x).cuda().unsqueeze(0).repeat(num_of_image, 1, 1, 1))
            print(f'Mean LPIPS: {LPIPS.mean()}')
            print(f'Mean PSNR: {psnr.mean()}')
            print(f'LPIPS: {LPIPS[:,0,0,0]}')
            print(f'PSNR: {psnr}')

            # Compute similarity
            # total_pair_wise_sim = 0.
            # xo_list = dino(images_DINO.cuda())
            # xo_list /= xo_list.norm(dim=-1, keepdim=True)
            # sim = (xo_list @ xo_list.T)
            # sim = sim - torch.diag(sim.diag())
            # total_pair_wise_sim += sim.sum() / (sim.shape[0] * (sim.shape[0] - 1))
            # print(total_pair_wise_sim)

            # Save results.
            output_path_img = f'{output_path}/{cfg.exp.img_id.split('.')[0]}/{cfg.algo.gamma}'
            if not os.path.exists(output_path_img):
                os.makedirs(output_path_img)

            for i in range(num_of_image):
                save_image(images_DINO[i], f'{output_path_img}/x_hat_{i}.png')
            save_image(image_grid, f'{output_path_img}/x_hat_grid.png')

            torch.cuda.empty_cache()
            logger.info(f'Done. You can fine the generated images in {output_path_img}')

        else:
            # We iterate over the data loader
            loader = build_loader(cfg)
            logger.info(f'Dataset size is {len(loader.dataset)}')


            for it, (x, y, info) in enumerate(loader):
                logger.info(f"Input image:{ info['index'][0]}")
                
                # If 256 size, upsample to 512
                if x.shape[-1] == 256:
                    upsample = nn.Upsample(scale_factor=2, mode='nearest') 
                    x = upsample(x).squeeze()

                # Get noisy measurement. TODO: Do a separate function
                x = preprocess(x)
                y_0 = H.H(x.unsqueeze(0).to("cuda"))
                y_0 = y_0 + torch.randn_like(y_0) * cfg.algo.sigma_x0    #?? what is it for???
                logger.info(f'Image loaded')

                # If save deg true, then we save the degradated image
                if cfg.exp.save_deg is True:
                    xo = postprocess(get_degreadation_image(y_0, H, cfg))
                    transform_pil_to_image = transforms.ToPILImage()
                    img = transform_pil_to_image(xo[0].cpu())
                    deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, 'x_deg.png')
                    save_image(xo[0].cpu(), deg_path)
                    logger.info(f'Degradated image y saved in {deg_path}')
                    
                # Prompt empty for stable diffusion
                prompt = ['']
                num_of_image = cfg.algo.n_particles
                prompt = prompt * num_of_image

                # Run the algorithm
                logger.info(f'Running setting with coeff: {cfg.algo.gamma}, \
                            lr: {cfg.algo.lr_x, cfg.algo.lr_z}, \
                            rho_reg: {cfg.algo.rho_reg}, \
                            w_t: {cfg.algo.w_t}')
                
                out = pipe.sample(prompt, 
                                generator=generator, 
                                num_inference_steps=cfg.exp.num_steps, 
                                gamma=cfg.algo.gamma, 
                                guidance_scale=cfg.algo.w, 
                                dino=dino, 
                                lr_x = cfg.algo.lr_x, 
                                lr_z = cfg.algo.lr_z, 
                                output_type='tensor',
                                H = H, 
                                y_0 = y_0, 
                                rho_reg=cfg.algo.rho_reg, 
                                w_t=cfg.algo.w_t, 
                                evol_path = evol_path)

                images_DINO = postprocess(out.images)
                list_images = [images_DINO.cpu()]
                image_grid = make_grid(images_DINO.cpu())

                # Compute metrics. TODO: separate function
                # PSNR
                mse = torch.mean((images_DINO - postprocess(x).cuda().unsqueeze(0).repeat(num_of_image, 1, 1, 1)) ** 2, dim=(1, 2, 3))
                psnr = 10 * torch.log10(1 / (mse + 1e-10))

                # LPIPS
                LPIPS = lpips_(images_DINO, postprocess(x).cuda().unsqueeze(0).repeat(num_of_image, 1, 1, 1))
                print(f'Mean LPIPS: {LPIPS.mean()}')
                print(f'Mean PSNR: {psnr.mean()}')
                print(f'LPIPS: {LPIPS[:,0,0,0]}')
                print(f'PSNR: {psnr}')

                # Compute similarity
                # total_pair_wise_sim = 0.
                # xo_list = dino(images_DINO.cuda())
                # xo_list /= xo_list.norm(dim=-1, keepdim=True)
                # sim = (xo_list @ xo_list.T)
                # sim = sim - torch.diag(sim.diag())
                # total_pair_wise_sim += sim.sum() / (sim.shape[0] * (sim.shape[0] - 1))
                # print(total_pair_wise_sim)

                # Save results.
                output_path_img = f'{output_path}/{info['index'][0].split('.')[0]}/{cfg.algo.gamma}'
                if not os.path.exists(output_path_img):
                    os.makedirs(output_path_img)

                for i in range(num_of_image):
                    save_image(images_DINO[i], f'{output_path_img}/x_hat_{i}.png')
                save_image(image_grid, f'{output_path_img}/x_hat_grid.png')

                torch.cuda.empty_cache()
                logger.info(f'Done. You can fine the generated images in {output_path_img}')
    else:
        raise NotImplementedError('Only RSD is implemented')


if __name__ == "__main__":
    main()
