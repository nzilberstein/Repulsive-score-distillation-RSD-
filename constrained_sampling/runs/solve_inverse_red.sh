gpu=1
algo="dps"
deg="deblur_motion"
num_steps=1000
sigma_y=0.001
n_particles=2
lr=0.25

python main_red_diff.py exp.load_img_id=True\
        algo=$algo algo.deg=$deg exp.num_steps=$num_steps algo.sigma_y=$sigma_y\
        algo.n_particles=$n_particles\
        algo.lr=$lr\
        exp.seed=0\
        +algo.mask_idx=35 \
        +exp.img_path="/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/FFHQ" \
        +exp.img_id="00005.png" \
        +exp.gpu=$gpu

# input="/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/IMAGENET/names.csv"
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


#         python main_red_diff.py exp.load_img_id=True\
#                         algo=$algo\
#                         algo.deg=$deg\
#                         exp.num_steps=$num_steps\
#                         algo.sigma_y=$sigma_y\
#                         algo.n_particles=$n_particles\
#                         exp.seed=10\
#                         +algo.mask_idx=$counter\
#                         +exp.img_path="/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/IMAGENET" \
#                         +exp.img_id="$last_field" \
#                         +exp.gpu=$gpu
            

# done < "$input"