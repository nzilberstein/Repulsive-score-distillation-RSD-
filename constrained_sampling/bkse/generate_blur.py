import argparse
import sys 
sys.path.append("/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling")
import cv2
import numpy as np
import os.path as osp
import torch
import utils.util as util
import yaml
from models.kernel_encoding.kernel_wizard import KernelWizard
import pdb

def main():
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(description="Kernel extractor testing")

    parser.add_argument("--image_path", action="store", help="image path", type=str, required=True)
    parser.add_argument("--yml_path", action="store", help="yml path", type=str, required=True)
    parser.add_argument("--save_path", action="store", help="save path", type=str, default=".")
    parser.add_argument("--num_samples", action="store", help="number of samples", type=int, default=1)

    args = parser.parse_args()

    image_path = args.image_path
    yml_path = args.yml_path
    num_samples = args.num_samples

    # Initializing mode
    with open(yml_path, "r") as f:
        opt = yaml.safe_load(f)["KernelWizard"]
        model_path = opt["pretrained"]
    model = KernelWizard(opt)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    HQ = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.0
    HQ = np.transpose(HQ, (2, 0, 1))
    HQ_tensor = torch.Tensor(HQ).unsqueeze(0).to(device).cuda()
    # HQ_tensor =HQ_tensor.repeat(4, 1, 1, 1)
    # pdb.set_trace()

    for i in range(num_samples):
        print(f"Sample #{i}/{num_samples}")
        with torch.no_grad():
            kernel = torch.randn((1, 512, 4, 4)).cuda() * 0.5#1.2
            LQ_tensor = model.adaptKernel(HQ_tensor, kernel)

        dst = osp.join(args.save_path, f"blur{i:03d}.png")
        print(dst)
        LQ_img = util.tensor2img(LQ_tensor)

        cv2.imwrite(dst, LQ_img)


main()
