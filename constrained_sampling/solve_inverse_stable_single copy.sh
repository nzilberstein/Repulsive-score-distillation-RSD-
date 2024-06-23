gpu=6
deg='inp_20ff'
num_steps=999
# rho_reg=0.05 #0.05
# w_t=0.0001     #0.0009, 0.05
# lr_x=0.9 #1
# lr_z=0.9

rho_reg=0.075      #0.05
    w_t=0.01   #0.0009, 0.01
    lr_x=0.4 #0.4
    lr_z=0.8 #0.8
sigma_x0=0.001
gamma=0

for w_t in 0.01
do
python main_rsd_diff.py exp.load_img_id=True\
                algo.deg=$deg\
                exp.num_steps=$num_steps\
                algo.rho_reg=$rho_reg\
                algo.w_t=$w_t\
                algo.lr_x=$lr_x\
                algo.lr_z=$lr_z\
                algo.sigma_x0=$sigma_x0\
                algo.gamma=$gamma\
                +exp.img_path="/home/nzilberstein/repository/constrained_sampling/_exp/input/imagenet/" \
                +exp.img_id="n01537544_indigo_bunting.JPEG" \
                +exp.gpu=$gpu
done
