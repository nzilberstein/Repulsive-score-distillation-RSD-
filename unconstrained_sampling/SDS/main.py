# make sure you're logged in with \`huggingface-cli login\`
import torch
import os
from torchvision.utils import make_grid, save_image
from PIL import Image
import os
import hydra
from hydra.core.hydra_config import HydraConfig
import lpips
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import tqdm

from models.stable_diffusion.diffusers import StableDiffusionParticleSDSPipeline, DDIMScheduler

from utils.torch_utils import seed_everything, get_logger, init_omega
from utils.functions import postprocess, preprocess

@hydra.main(version_base="1.2", config_path="_configs", config_name="ddrmpp")
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
    output_path = os.path.join(cfg.exp.root, cfg.exp.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    evol_path = os.path.join(cfg.exp.root, cfg.exp.evol_path)

    # Get logger
    logger = get_logger(name="main", cfg=cfg)
    logger.info(f'Experiment name is {"unconstrained RSD"}')

    # Load model
    pipe = StableDiffusionParticleSDSPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
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

    ### UP TO THIS POINT, EVERYTHING IS THE SAME EITHER FOR STABLE DIFFUSION OR OTHER DIFFUSION MODELS.

    if cfg.exp.load_text is True:
        raise NotImplementedError
    else:
        # TODO: FOllow same logic of inverse: if input, handle alone, otherwise load from file
        # Load files
        input_path = os.path.join(cfg.exp.root, cfg.exp.csv_path)
        df = pd.read_csv(input_path)
        all_text = list(df['caption'])
        all_text = all_text[: cfg.algo.max_cnt]
        # index_list = np.arange(len(all_text))

        # Generate samples
        for cnt, mini_batch in enumerate(tqdm.tqdm(all_text)):
            text = [str(mini_batch)]
            num_of_image = cfg.algo.n_particles
            text = text * num_of_image
            # TODO: CHANGE LOGIG OF PIPE
            out = pipe.sample(text, 
                              generator=generator, 
                              num_inference_steps=cfg.exp.num_steps, 
                              gamma= cfg.algo.gamma, 
                              guidance_scale=cfg.algo.w, 
                              dino=dino, 
                              lr = cfg.algo.lr, 
                              output_type='tensor',
                              evol_path=evol_path)
            
            images_DINO = postprocess(out.images)
            image_grid = make_grid(images_DINO.cpu())

            # Save results.
            output_path_img = f'{output_path}/{cnt}'
            if not os.path.exists(output_path_img):
                os.mkdir(output_path_img)
            for i in range(num_of_image):
                save_image(images_DINO[i], f'{output_path_img}/x_hat_{i}.png')
            save_image(image_grid, f'{output_path_img}/x_hat_grid.png')

if __name__ == '__main__':
    # args = parser.parse_args()
    main()