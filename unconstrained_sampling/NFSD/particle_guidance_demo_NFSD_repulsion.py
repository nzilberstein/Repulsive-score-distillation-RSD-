# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionParticleNFSDPipeline, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, SDEScheduler, EulerDiscreteScheduler
import torch
import argparse
import os
from torchvision.utils import make_grid, save_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import lpips
import shutil

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--coeff', type=float, default=0.)
parser.add_argument('--t_repul', type=int, default=1000)
args = parser.parse_args()
        

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

# torch.cuda.set_device(7)``

pipe = StableDiffusionParticleNFSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
# pipe = StableDiffusionParticleSDSPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# load dino 
dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to("cuda")

# prompt = ["a blue bicycle parked by a metal gate."]
# prompt = ["An astronaut riding a horse"]
prompt = ["A metal lying buddah"]
neg_prompt = ['unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy']
num_of_image = 4
prompt = prompt * num_of_image
neg_prompt = neg_prompt * num_of_image


seed = 1 # Seed 6 for cookies, 0 for buddah
# number of sampling steps
# steps = 999

#1000 steps
# steps = 999
# lr = 0.01

# 500 steps
steps = 999
lr = 0.01 #0.015 for 500 steps, 0.01 for 1ksteps
# guidance scale
w = 7.5

torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.random.manual_seed(seed)
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


generator = torch.Generator(device="cuda").manual_seed(seed)
# coefficient for the particle guidance scale 
coeff_list = [100, 200, 500] #,  300, 500, 900] #0, 2, 5 similar
# coeff_list = [700, 900]
# lr_list = [0.015, 0.02]#, 0.05]
t_repul_list = [900, 800, 500, 200]
# coeff = 1000
it = 1

# for lr in lr_list:
# Clean folder in /home/nicolas/RED-diff_stable/_exp/nfsd_stable_lpips before starting
dir_evol = "/home/nzilberstein/red_diff_stable/RED-diff_stable/_exp/nfsd_stable_dino"
if os.path.exists(dir_evol):
    shutil.rmtree(dir_evol)
os.makedirs(dir_evol)

neg_prompt = ['unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy']
neg_prompt = neg_prompt * num_of_image
print(args.coeff, num_of_image)

# out = pipe.dino(prompt, generator=generator, num_inference_steps=steps, coeff=coeff, guidance_scale=w, dino=dino, output_type='tensor')
out = pipe.dino_NFSD(prompt, generator=generator, num_inference_steps=steps, coeff=args.coeff, guidance_scale=w, dino=dino,  lr = lr, output_type='tensor', negative_prompt=neg_prompt, t_repul = args.t_repul) #0.4 for ADAM
# out = pipe(prompt, generator=generator, num_inference_steps=steps, coeff=args.coeff, guidance_scale=w, svgd=True,  lr = lr, output_type='tensor', negative_prompt=neg_prompt, t_repul = args.t_repul) #0.4 for ADAM
# out = pipe.dino_NFSD(prompt, generator=generator, num_inference_steps=steps, coeff=coeff, guidance_scale=w, dino=dino, lr = lr, output_type='tensor', negative_prompt = neg_prompt)
images_DINO = out.images
images_DINO = torch.from_numpy(images_DINO).permute(0, 3, 1, 2)
save_image(images_DINO, f'/home/nzilberstein/red_diff_stable/RED-diff_stable/_exp/nfsd_outputs_paper/buddah_trepul_{args.t_repul}_coeff_{args.coeff}.pdf')
image_grid = make_grid(images_DINO)
# plt.figure(it, figsize=(20, 20))
# plt.imshow(image_grid.permute(1, 2, 0))
# plt.show()

total_pair_wise_sim = 0.
xo_list = dino(images_DINO.cuda())
# del dino
xo_list /= xo_list.norm(dim=-1, keepdim=True)
# calculate the cosine similarity
sim = (xo_list @ xo_list.T)
# set the diagonal to be 0
sim = sim - torch.diag(sim.diag())
total_pair_wise_sim += sim.sum() / (sim.shape[0] * (sim.shape[0] - 1))
print(total_pair_wise_sim)