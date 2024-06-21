# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionParticleSDSPipeline, DDIMScheduler, EulerDiscreteScheduler
import torch
from coco_data_loader import text_image_pair
from PIL import Image
import os
import pandas as pd
import argparse
import torch.nn as nn
from torch_utils import distributed as dist
import numpy as np
import tqdm
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=30, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--w', type=float, default=7.5)
parser.add_argument('--s_noise', type=float, default=1.)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--max_cnt', type=int, default=5000, help='number of maximum geneated samples')
parser.add_argument('--save_path', type=str, default='/home/nzilberstein/red_diff_stable/RED-diff_stable/particle_guidance/stable_diffusion/generated_images_SDS')
parser.add_argument('--scheduler', type=str, default='DDIM')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument('--second', action='store_true', default=False, help='second order ODE')
parser.add_argument('--sigma', action='store_true', default=False, help='use sigma')
parser.add_argument('--coeff', type=float, default=0.)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dino', action='store_true', default=False, help='use dino')
parser.add_argument('--csv_path', type=str, default='/home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/subset_object_10.csv')
parser.add_argument('--n_particles', type=int, default=4)
parser.add_argument('--gpu', type=int, default=0)

def main(args):
    ####
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"

    torch.cuda.set_device(args.gpu)

    # Generator for stable diffusion
    seed_everything(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(cfg.exp.seed)

    # Load config file
    cwd = HydraConfig.get().runtime.output_dir
    cfg = init_omega(cfg, cwd)

    df = pd.read_csv(args.csv_path)
    all_text = list(df['caption'])
    all_text = all_text[: args.max_cnt]
    # num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
    # all_batches = np.array_split(np.array(all_text), num_batches)
    # rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    # all_batches_index = np.array_split(index_list, num_batches)
    # rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]

    ##### load stable diffusion models #####
    pipe = StableDiffusionParticleSDSPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    # dist.print0("default scheduler config:")
    # dist.print0(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    if args.dino:
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)

    ##### setup save configuration #######
    if args.name is None:
        save_dir = os.path.join(args.save_path,
                                f'steps_{args.steps}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_dino_{args.dino}_coeff_{args.coeff}_lr_{args.lr}')
    else:
        save_dir = os.path.join(args.save_path,
                                f'steps_{args.steps}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_dino_{args.dino}_coeff_{args.coeff}_lr_{args.lr}_name_{args.name}')

    # dist.print0("save images to {}".format(save_dir))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print(all_text[0])

    ## generate images ##
    assert args.bs == 1
    for cnt, mini_batch in enumerate(tqdm.tqdm(all_text)):
        # torch.distributed.barrier()
        if len(mini_batch) == 0:
            continue
        text = [str(mini_batch)]
        # generate four images using the same text
        text = text * args.n_particles
        # print(text)
        if args.dino:
            out = pipe.dino_SDS(text, generator=generator, num_inference_steps=args.steps, coeff= args.coeff, guidance_scale=args.w, dino=dino, lr = args.lr, output_type='tensor')
        else:
            out, _ = pipe(text, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w, restart=args.restart,
                        second_order=args.second, dist=dist, S_noise=args.s_noise, coeff=args.coeff)
        image = out.images

        # for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
        path = os.path.join(save_dir, f'{cnt}')
        if not os.path.exists(path):
            os.mkdir(path)
        for i in range(args.n_particles):
            save_image(torch.from_numpy(image[i]).permute(2, 0, 1), os.path.join(path, f'{i}.png'))
            # torch.from_numpy(image[i]).permute(2, 0, 1).save(os.path.join(path, f'{i}.png'))

    # Done.
    # torch.distributed.barrier()
    # if dist.get_rank() == 0:
    #     d = {'caption': all_text}
    #     df = pd.DataFrame(data=d)
    #     df.to_csv(os.path.join(save_dir, 'subset.csv'))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)