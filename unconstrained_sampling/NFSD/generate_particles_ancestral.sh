w=7.5
n_particles=4
steps=30
coeff=0
lr=0.01
gpu=6

for coeff in 20
do 
python generate_particles_ancestral.py \
        --csv_path /home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/subset_object_80.csv\
        --w $w --n_particles $n_particles --steps $steps --coeff $coeff --gpu $gpu --dino

done