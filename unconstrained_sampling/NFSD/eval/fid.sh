# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/0.0
# dest_folder=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/FID/0.0

# python eval_FID_score_particle_SDS.py --dir_path $dir_path --dest_folder $dest_folder

# python -m pytorch_fid /home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/val2014_resize $dest_folder

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/1e-10
# dest_folder=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/FID/1e-10

# python eval_FID_score_particle_SDS.py --dir_path $dir_path --dest_folder $dest_folder

# python -m pytorch_fid /home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/val2014_resize $dest_folder

dir_path=/home/nzilberstein/red_diff_stable/RED-diff_stable/particle_guidance/stable_diffusion/generated_images_SDS/steps_999_w_100.0_second_False_seed_6_dino_True_coeff_0.0_lr_0.01
dest_folder=/home/nzilberstein/red_diff_stable/RED-diff_stable/particle_guidance/stable_diffusion/generated_images_SDS/FID

python eval_FID_score_particle_SDS.py --dir_path $dir_path --dest_folder $dest_folder

python -m pytorch_fid /home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/val2014_resize $dest_folder

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/10.0
# dest_folder=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/FID/10.0

# python eval_FID_score_particle_SDS.py --dir_path $dir_path --dest_folder $dest_folder

# python -m pytorch_fid /home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/val2014_resize $dest_folder

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/20.0
# dest_folder=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/FID/20.0

# python eval_FID_score_particle_SDS.py --dir_path $dir_path --dest_folder $dest_folder

# python -m pytorch_fid /home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/val2014_resize $dest_folder

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/30.0
# dest_folder=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/FID/30.0

# python eval_FID_score_particle_SDS.py --dir_path $dir_path --dest_folder $dest_folder

# python -m pytorch_fid /home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/val2014_resize $dest_folder


# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/40.0
# dest_folder=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/FID/40.0

# python eval_FID_score_particle_SDS.py --dir_path $dir_path --dest_folder $dest_folder

# python -m pytorch_fid /home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/val2014_resize $dest_folder


# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/ancestral
# dest_folder=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/FID/ancestral

# python eval_FID_score_particle_SDS.py --dir_path $dir_path --dest_folder $dest_folder

# python -m pytorch_fid /home/nzilberstein/red_diff_stable/RED-diff_stable/datacoco/val2014_resize $dest_folder
