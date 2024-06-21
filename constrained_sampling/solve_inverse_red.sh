gpu=1
algo="dps"
deg="phase_retrieval"
num_steps=1000
grad_term_weight=0.5
sigma_y=0.001

python main_red_diff.py exp.load_img_id=True\
        algo=$algo algo.deg=$deg exp.num_steps=$num_steps algo.grad_term_weight=$grad_term_weight\
        +exp.img_path="/home/nzilberstein/repository/constrained_sampling/_exp/input/FFHQ" \
        +exp.img_id="00004.png" \
        +exp.gpu=$gpu

# python main_red_diff.py exp.load_img_id=False \
#                         algo=$algo algo.deg=$deg exp.num_steps=$num_steps algo.grad_term_weight=$grad_term_weight\
#                         +exp.gpu=$gpu

# 