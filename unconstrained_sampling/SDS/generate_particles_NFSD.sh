w=10
n_particles=4
steps=999
coeff=0
lr=0.01
gpu=2

for coeff in 0
do 
python generate_particles_NFSD.py \
        --csv_path /home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/subset_3_bis.csv\
        --save_path '/home/nzilberstein/red_diff_stable/RED-diff_stable/particle_guidance/stable_diffusion/generated_images_NFSD_change_CFG'\
        --w $w --dino --n_particles $n_particles --steps $steps --coeff $coeff --lr $lr --gpu $gpu

done