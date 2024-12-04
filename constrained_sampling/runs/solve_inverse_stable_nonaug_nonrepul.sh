gpu=1
algo='rsd_stable_nonaug'
deg='inp_large_box'
num_steps=500

# Super-resolution
# rho_reg=0.05     #0.05
#     w_t=0.01   #0.0009, 0.01
#     lr_x=0.4 #0.4
#     lr_z=0.6 #0.8
# sigma_x0=0.001
# gamma=0
# n_particles=2
# dino_flag=False

# Phase
# rho_reg=0.07     #0.05
#     w_t=0.02 #0.15   #0.0009, 0.01
#     lr_x=0.6 #0.4
#     lr_z=0.6 #1.2 #0.8
# sigma_x0=0
# gamma=0
# n_particles=2
# dino_flag=False

# HDR
# rho_reg=0.2     #0.09
#     w_t=0.001      #0.001
#     lr_x=0.1      #0.4
#     lr_z=0.2      #0.8
# sigma_x0=0.001
# gamma=30
# n_particles=2
# dino_flag=False

# Motion deblurring
# rho_reg=0.009     #0.09
#     w_t=0.0001      #0.001
#     lr_x=0.3      #0.4
#     lr_z=0.3      #0.8
# sigma_x0=0.001
# gamma=0
# n_particles=2
# dino_flag=False

# Random inp
# rho_reg=0.09      #0.05
#     w_t=0.001   #0.0009, 0.01
#     lr_x=0.4 #0.4
#     lr_z=0.8 #0.8
# sigma_x0=0.001
# gamma=0

# Half face
rho_reg=0.075      #0.3
w_t=0.1 #0.14
lr_x=0.4
lr_z=0.6
sigma_x0=0.001
gamma=0
n_particles=4
dino_flag=False
sigma_break=999


python main_rsd_diff.py exp.load_img_id=False \
                algo.deg=$deg\
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
                algo.name=$algo\
                exp.seed=10\
               +exp.gpu=$gpu



# input="/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/input.csv"
# counter=0

# while IFS= read -r line
# do
#         counter=$((counter+1))

#         # Skip the first line
#         if [ "$counter" -eq 1 ]; then
#          continue
#         fi

#         # Use awk to extract the last field from each line
#         last_field=$(echo "$line" | awk -F ',' '{print $NF}')
        
#         echo "$last_field"


#         python main_rsd_diff.py exp.load_img_id=True\
#                         algo.deg=$deg\
#                         algo.name=$algo\
#                         exp.num_steps=$num_steps\
#                         algo.rho_reg=$rho_reg\
#                         algo.w_t=$w_t\
#                         algo.lr_x=$lr_x\
#                         algo.lr_z=$lr_z\
#                         algo.sigma_x0=$sigma_x0\
#                         algo.gamma=$gamma\
#                         algo.n_particles=$n_particles\
#                         algo.dino_flag=$dino_flag\
#                         exp.seed=1\
#                         +exp.img_path="/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/FFHQ" \
#                         +exp.img_id="$last_field" \
#                         +exp.gpu=$gpu
            

# done < "$input"