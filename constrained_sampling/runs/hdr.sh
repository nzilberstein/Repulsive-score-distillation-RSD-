gpu=1
algo='rsd_stable'
deg='hdr'
num_steps=999

rho_reg=0.2 #0.13     #0.09
    w_t=0.008 #0.02      #0.001
    lr_x=0.1      #0.4
    lr_z=0.2    #0.8
sigma_x0=0.001

gamma=0
n_particles=1
dino_flag=False
sigma_break=999

python main_rsd_diff.py exp.load_img_id=True\
                algo.deg=$deg\
                algo.name=$algo\
                exp.num_steps=$num_steps\
                algo.rho_reg=$rho_reg\
                algo.w_t=$w_t\
                algo.lr_x=$lr_x\
                algo.lr_z=$lr_z\
                algo.sigma_x0=$sigma_x0\
                algo.gamma=$gamma\
                algo.n_particles=$n_particles\
                algo.dino_flag=$dino_flag\
                algo.sigma_break=$sigma_break\
                exp.seed=1\
                +algo.mask_idx=15 \
                +exp.img_path="/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/FFHQ" \
                +exp.img_id="00005.png" \
                +exp.gpu=$gpu
done