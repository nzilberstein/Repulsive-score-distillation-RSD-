gpu=0
algo='rsd_stable'
deg='inp_20ff'
num_steps=200

# Phase retrieval
# rho_reg=0.07     #0.05
#     w_t=0.02 #0.15   #0.0009, 0.01
#     lr_x=0.6 #0.4
#     lr_z=0.6 #1.2 #0.8
# sigma_x0=0.001
# n_particles=6
# gamma=0
# dino_flag=False
# sigma_break=999


# HDR
# rho_reg=0.2     #0.09
#     w_t=0.001      #0.001
#     lr_x=0.1      #0.4
#     lr_z=0.2      #0.8
# sigma_x0=0.001


# Inp small box
# rho_reg=0.08     #0.05
#     w_t=0.009   #0.0009, 0.01
#     lr_x=0.3 #0.4
#     lr_z=0.3 #0.8
# sigma_x0=0.001

#Motion deblurring
# rho_reg=0.009     #0.09
#     w_t=0.0001      #0.001
#     lr_x=0.3      #0.4
#     lr_z=0.3      #0.8
# sigma_x0=0.001
# gamma=0

# Free mask
rho_reg=0.1      #0.3
    w_t=0.15      #0.15 for 6 particles and 300 iter
    lr_x=0.4     #0.4 for 6 part, 1.6 for 2 particles
    lr_z=1     #1 for 6 part, 4 for 2 particles
sigma_x0=0.001
gamma=150
n_particles=4
dino_flag=True
sigma_break=999

# half face
# rho_reg=0.075      #0.3
# w_t=0.13
# lr_x=0.4
# lr_z=0.6
# sigma_x0=0.001
# gamma=0
# n_particles=4
# dino_flag=False
# sigma_break=999

# for gamma in 10 30
# do
# for lr_z in 0.6 0.8 1.0
# do
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
                exp.seed=10\
                +algo.mask_idx=0 \
                +exp.img_path="/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/FFHQ" \
                +exp.img_id="00005.png" \
                +exp.gpu=$gpu
# done
# done 

                # +exp.img_path="/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/IMAGENET/" \
                # +exp.img_id="n01440764.png" \


# gpu=1
# algo='rsd_stable'
# deg='phase_retrieval'
# num_steps=999

# rho_reg=0.1     #0.01
#     w_t=0.01      #0.001
#     lr_x=0.2    #0.2 for 2 particles
#     lr_z=0.25   #0.25 for 2 particles
# sigma_x0=0.001

# # rho_reg=0.02      #0.09
# #     w_t=0.01      #0.001
# #     lr_x=0.2      #0.4
# #     lr_z=0.2      #0.8
# # sigma_x0=0.001
# gamma=0
# n_particles=2
# dino_flag=False

# # 0.001
# python main_rsd_diff.py exp.load_img_id=True\
#                 algo.deg=$deg\
#                 algo.name=$algo\
#                 exp.num_steps=$num_steps\
#                 algo.rho_reg=$rho_reg\
#                 algo.w_t=$w_t\
#                 algo.lr_x=$lr_x\
#                 algo.lr_z=$lr_z\
#                 algo.sigma_x0=$sigma_x0\
#                 algo.gamma=$gamma\
#                 algo.n_particles=$n_particles\
#                 algo.dino_flag=$dino_flag\
#                 exp.seed=100\
#                 +exp.img_path="/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/FFHQ" \
#                 +exp.img_id="00005.png" \
#                 +exp.gpu=$gpu

# # gpu=1
# # algo='rsd_stable'
# # deg='deblur_motion'
# # num_steps=999

# # rho_reg=0.01     #0.01
# #     w_t=0.008      #0.001
# #     lr_x=0.4    #0.2 for 2 particles
# #     lr_z=0.4    #0.25 for 2 particles
# # sigma_x0=0.001

# # # rho_reg=0.02      #0.09
# # #     w_t=0.01      #0.001
# # #     lr_x=0.2      #0.4
# # #     lr_z=0.2      #0.8
# # # sigma_x0=0.001
# # gamma=0
# # n_particles=4
# # dino_flag=False

# # # 0.001
# # for w_t in 0.001
# # do
# # python main_rsd_diff.py exp.load_img_id=True\
# #                 algo.deg=$deg\
# #                 algo.name=$algo\
# #                 exp.num_steps=$num_steps\
# #                 algo.rho_reg=$rho_reg\
# #                 algo.w_t=$w_t\
# #                 algo.lr_x=$lr_x\
# #                 algo.lr_z=$lr_z\
# #                 algo.sigma_x0=$sigma_x0\
# #                 algo.gamma=$gamma\
# #                 algo.n_particles=$n_particles\
# #                 algo.dino_flag=$dino_flag\
# #                 exp.seed=100\
# #                 +exp.img_path="/home/nicolas/dataset/imagenet-root/imagenet/val/n01514668/" \
# #                 +exp.img_id="ILSVRC2012_val_00000329.JPEG" \
# #                 +exp.gpu=$gpu
# # done
