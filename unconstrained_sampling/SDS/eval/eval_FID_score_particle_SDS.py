import torch
from PIL import Image
import argparse
import os
import numpy as np
from torchvision.utils import make_grid, save_image
from torchvision.transforms import transforms
from pytorch_fid import fid_score

import torch.nn as nn

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--dir_path', type=str, default='/home/nzilberstein/Inverse/_exp/input')
parser.add_argument('--dest_folder', type=str, default='/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/FID/ancestral')
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()

torch.cuda.set_device(6)

# Define paths to the two folders
# folder2 = '/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/0.0_lr003'

# torch.manual_seed(args.seed)                            
# np.random.seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# os.environ['PYTHONHASHSEED'] = str(args.seed)
# torch.random.manual_seed(args.seed)

dest_folder = args.dest_folder
# dest_folder = '/home/nzilberstein/red_diff_stable/RED-diff_stable/particle_guidance/stable_diffusion/generated_images_ancestral/FID'
# dest_folder = '/home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/val2014'
# Calculate FID score
it = 0
for item in os.listdir(args.dir_path):
    # Check if the item is a directory (folder)
    if item.isnumeric():
        print(item)
        list_imgs = os.listdir(os.path.join(args.dir_path, item))
        idx_img = np.random.choice(len(os.listdir(os.path.join(args.dir_path, item))))
        img_true = Image.open(os.path.join(args.dir_path, item, list_imgs[idx_img])).convert('RGB')

        img_torch = transforms.ToTensor()(img_true)

        # Create folder if does not exist
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        save_image(img_torch, f'{dest_folder}/{it}.png')
    it = it + 1



# Resize
# list_imgs = os.listdir(args.dir_path)
# for item in list_imgs:
#     # Check if the item is a directory (folder)
#     # list_imgs = os.listdir(args.dir_path)
#     print(item)
#     img_true = Image.open(os.path.join(args.dir_path, item)).convert('RGB')

#     img_torch = transforms.ToTensor()(img_true)
#     upsample = nn.Upsample(scale_factor=2, mode='nearest') 
#     img_torch = upsample(img_torch.unsqueeze(0)).squeeze()

#     # # Create folder if does not exist
#     # if not os.path.exists(dest_folder):
#     #     os.makedirs(dest_folder)

#     save_image(img_torch, f'/home/nzilberstein/Inverse/_exp/input/FFHQ_512/{item}')
#     it = it + 1