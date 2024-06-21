# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionParticlePipeline, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, SDEScheduler, EulerDiscreteScheduler
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
parser.add_argument('--save_path', type=str, default='/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/ancestral')
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
parser.add_argument('--t_repul', type=int, default=999)
parser.add_argument('--seed', type=int, default=6)

args = parser.parse_args()

# dist.init()

# if dist.get_rank() == 0:
#     if not os.path.exists(args.save_path):
#         os.mkdir(args.save_path)
# torch.distributed.barrier()


# dist.print0('Args:')
# for k, v in sorted(vars(args).items()):
    # dist.print0('\t{}: {}'.format(k, v))
# define dataset / data_loader

torch.cuda.set_device(args.gpu)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.random.manual_seed(args.seed)

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
pipe = StableDiffusionParticlePipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
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
    print(text)
    # generate four images using the same text
    text = text * args.n_particles
    # print(text)

    print(1)
    out = pipe(text, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w, output_type='tensor')
    # out = pipe.dino(text, generator=generator, num_inference_steps=args.steps, coeff=args.coeff, guidance_scale=args.w, dino=dino, output_type='tensor')
    image = out.images

    # for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
    path = os.path.join(save_dir, f'{cnt + 37}')
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(args.n_particles):
        save_image(torch.from_numpy(image[i]).permute(2, 0, 1), os.path.join(path, f'{i}.png'))
        # torch.from_numpy(image[i]).permute(2, 0, 1).save(os.path.join(path, f'{i}.png'))

    if cnt == args.max_cnt:
        break

# Done.
# torch.distributed.barrier()
# if dist.get_rank() == 0:
#     d = {'caption': all_text}
#     df = pd.DataFrame(data=d)
#     df.to_csv(os.path.join(save_dir, 'subset.csv'))