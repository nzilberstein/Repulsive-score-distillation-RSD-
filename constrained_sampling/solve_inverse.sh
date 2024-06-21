gpu=2
deg='inp_small_box'
num_steps=999
rho_reg=0.05 #0.05
w_t=0.1 #0.01, 0.005 for box
lr_x=0.5 #1
lr_z=0.5

# 0.05 good
# for w_t in 0.05
# do
python main.py exp.load_img_id=True\
            algo.deg=$deg\
            exp.num_steps=$num_steps\
            algo.rho_reg=$rho_reg\
            algo.w_t=$w_t\
            algo.lr_x=$lr_x\
            algo.lr_z=$lr_z\
            +exp.img_path="/home/nzilberstein/Inverse/_exp/input/FFHQ" \
            +exp.img_id="00003.png" \
            +exp.gpu=$gpu
# done

# python main.py exp.load_img_id=False \
#                 algo.deg=$deg\
#                 exp.num_steps=$num_steps\
#                 algo.rho_reg=$rho_reg\
#                 algo.w_t=$w_t\
#                 algo.lr_x=$lr_x\
#                 algo.lr_z=$lr_z\
#                 +exp.gpu=$gpu

