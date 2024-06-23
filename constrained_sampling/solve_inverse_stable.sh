gpu=1
deg='inp_random'
num_steps=999

# Random inp
rho_reg=0.09      #0.05
    w_t=0.001   #0.0009, 0.01
    lr_x=0.4 #0.4
    lr_z=0.8 #0.8
sigma_x0=0.001
gamma=0

# Half face
# rho_reg=0.075      #0.05
#     w_t=0.01   #0.0009, 0.01
#     lr_x=0.4 #0.4
#     lr_z=0.8 #0.8
# sigma_x0=0.001
# gamma=0

python main_rsd_diff.py exp.load_img_id=False \
                algo.deg=$deg\
                exp.num_steps=$num_steps\
                algo.rho_reg=$rho_reg\
                algo.w_t=$w_t\
                algo.lr_x=$lr_x\
                algo.lr_z=$lr_z\
                algo.sigma_x0=$sigma_x0\
                algo.gamma=$gamma\
               +exp.gpu=$gpu

