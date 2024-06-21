import torch
from PIL import Image
import open_clip
from coco_data_loader import text_image_pair
import argparse
from tqdm import tqdm
import clip
import aesthetic_score
import os
import statistics

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=50, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--max_cnt', type=int, default=10000, help='number of maximum geneated samples')
parser.add_argument('--csv_path', type=str, default='/home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco')
parser.add_argument('--csv_file', type=str, default="subset_partial.csv")
parser.add_argument('--dir_path', type=str, default='/home/nicolas/RED-diff_stable/particle_guidance/stable_diffusion/generated_images_NFSD/steps_500_w_7.5_second_False_seed_6_dino_True_coeff_500.0_lr_0.012')
parser.add_argument('--scheduler', type=str, default='DDPM')
parser.add_argument('--prompt', type=str, default='a photo of an astronaut riding a horse on mars')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.gpu)

print("path:", args.dir_path)
# define dataset / data_loader
args.csv_path = os.path.join(args.csv_path, args.csv_file)
text2img_dataset = text_image_pair(dir_path=args.dir_path, csv_path=args.csv_path, group=True)
text2img_loader = torch.utils.data.DataLoader(dataset=text2img_dataset, batch_size=args.bs, shuffle=False)

print("total length:", len(text2img_dataset))
model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
model2, _ = clip.load("ViT-L/14", device='cuda')  #RN50x64
model = model.cuda().eval()
model2 = model2.eval()
tokenizer = open_clip.get_tokenizer('ViT-g-14')

model_aes = aesthetic_score.MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
s = torch.load("./clip-refs/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
model_aes.load_state_dict(s)
model_aes.to("cuda")
model_aes.eval()

dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').to("cuda")
# text = tokenizer(["a horse", "a dog", "a cat"])
cnt = 0.
total_clip_score = 0.
total_aesthetic_score = 0.
total_pair_wise_sim = 0.

clip_score_list = []
aesthetic_score_list = []
pair_wise_sim_list = []

cnt_iter = 0
with torch.no_grad(), torch.cuda.amp.autocast():
    for idx, (image,  text, dino_image) in tqdm(enumerate(text2img_loader)):
        print(text)
        image = image.cuda().float().squeeze(0)
        # print(image.shape, dino_image.shape)
        dino_image = dino_image.cuda().float().squeeze(0)
        # print(dino_image.shape)
        text = text * dino_image.shape[0]
        text = tokenizer(text).cuda()
        image_features = model.encode_image(image).float()
        text_features = model.encode_text(text).float()
        # (bs, 1024)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # DINO pairwise cosine similarity
        dino_features = dino(dino_image)
        dino_features /= dino_features.norm(dim=-1, keepdim=True)
        # calculate the pairwise cosine similarity between image_features
        sim = (dino_features @ dino_features.T)
        # set the diagonal to be 0
        sim = sim - torch.diag(sim.diag())
        total_pair_wise_sim += sim.sum() / (sim.shape[0] * (sim.shape[0] - 1))
        pair_wise_sim_list.append(sim.sum() / (sim.shape[0] * (sim.shape[0] - 1)))

        del sim
        del dino_features
        #############################

        text_features /= text_features.norm(dim=-1, keepdim=True)

        total_clip_score += (image_features * text_features).sum()
        clip_score_list.append((image_features * text_features).sum(1).mean())

        image_features = model2.encode_image(image)
        im_emb_arr = aesthetic_score.normalized(image_features.cpu().detach().numpy())
        aes_score = model_aes(torch.from_numpy(im_emb_arr).to(image.device).type(torch.cuda.FloatTensor))

        total_aesthetic_score += aes_score.sum()
        aesthetic_score_list.append(aes_score.mean())
        # print(statistics.median(aesthetic_score_list), )
        cnt += len(image)
        cnt_iter += 1

        if cnt >= args.max_cnt:
            break


print("Average pairwise similarity :", total_pair_wise_sim.item() / cnt_iter, "std:", torch.std(torch.tensor(pair_wise_sim_list)))
print("Average ClIP score :", total_clip_score.item() / cnt, "std:", torch.std(torch.tensor(clip_score_list)))
print("Average Aesthetic score :", total_aesthetic_score.item() / cnt, "std:", torch.std(torch.tensor(aesthetic_score_list)))
# print("Median Aesthetic score :", statistics.median(aesthetic_score_list), "std:", torch.std(torch.tensor(aesthetic_score_list)))