import torch
from PIL import Image
import open_clip
from coco_data_loader import text_image_pair
import argparse
from tqdm import tqdm
import clip
import aesthetic_score
import os

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=50, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--max_cnt', type=int, default=100, help='number of maximum geneated samples')
parser.add_argument('--dir_path_2', type=str, default='/home/nicolas/RED-diff_stable/particle_guidance/stable_diffusion/datacoco')
parser.add_argument('--dir_path_1', type=str, default='/home/nicolas/prolific_dreamer2d/generated_images_small')
parser.add_argument('--scheduler', type=str, default='DDPM')
parser.add_argument('--prompt', type=str, default='a photo of an astronaut riding a horse on mars')
parser.add_argument('--gpu', type=int, default=2)
args = parser.parse_args()

torch.cuda.set_device(args.gpu)

from pytorch_fid import fid_score
import os

# Define paths to the two folders
folder1 = '/path/to/first/folder'
folder2 = '/path/to/second/folder'

# Calculate FID score
fid = fid_score.calculate_fid_given_paths([folder1, folder2], batch_size=50, cuda=True)

print("FID Score:", fid)