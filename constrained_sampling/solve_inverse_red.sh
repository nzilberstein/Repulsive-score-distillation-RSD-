gpu=4
algo="reddiff"
deg="inp_20ff"
num_steps=1000
grad_term_weight=0.5
sigma_y=0.001

python main_red_diff.py exp.load_img_id=True\
        algo=$algo algo.deg=$deg exp.num_steps=$num_steps algo.grad_term_weight=$grad_term_weight\
        +exp.img_path="/home/nzilberstein/repository/constrained_sampling/_exp/input/imagenet" \
        +exp.img_id="n01537544_indigo_bunting.JPEG" \
        +exp.gpu=$gpu

# python main_red_diff.py exp.load_img_id=False \
#                         algo=$algo algo.deg=$deg exp.num_steps=$num_steps algo.grad_term_weight=$grad_term_weight\
#                         +exp.gpu=$gpu

# 