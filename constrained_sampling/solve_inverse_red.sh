gpu=1
algo="dps"
deg="inp_large_box"
num_steps=999
grad_term_weight=0.25
sigma_y=0.001
n_particles=2

python main_red_diff.py exp.load_img_id=False\
        algo=$algo algo.deg=$deg exp.num_steps=$num_steps algo.grad_term_weight=$grad_term_weight algo.sigma_y=$sigma_y\
        algo.n_particles=$n_particles\
        exp.seed=0\
        +exp.img_path="/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/FFHQ" \
        +exp.img_id="00000.png" \
        +exp.gpu=$gpu

# python main_red_diff.py exp.load_img_id=False \
#                         algo=$algo algo.deg=$deg exp.num_steps=$num_steps algo.grad_term_weight=$grad_term_weight\
#                         +exp.gpu=$gpu

# 