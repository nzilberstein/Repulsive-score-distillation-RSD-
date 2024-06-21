# Specify the path to the parent folder
# parent_folder="/home/nzilberstein/red_diff_stable/RED-diff_stable/particle_guidance/stable_diffusion/generated_images_ancestral/steps_30_w_7.5_second_False_seed_6_dino_False_coeff_42.0_lr_0.01"

# # Loop over each directory inside the parent folder
# for folder in "$parent_folder"/*; do
#     # Check if the item is a directory
#     if [ -d "$folder" ]; then
#         # If it's a directory, print its name or perform any other operations
#         echo "Folder: $folder"
#     fi
# done

# Replace "/path/t

gpu=0
# csv_file="subset_object_21.csv"
# csv_file="subset_3_bis.csv"
csv_file="subset_partial.csv"

# This corresponds to CFG 1
# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/0.0_CFG1
# python eval_clip_score_particle_SDS.py --gpu $gpu --dir_path $dir_path --csv_file $csv_file

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/0.0_CFG4
# python eval_clip_score_particle_SDS.py --gpu $gpu --dir_path $dir_path --csv_file $csv_file

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/10.00001
# python eval_clip_score_particle_SDS.py --gpu $gpu --dir_path $dir_path --csv_file $csv_file

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/0.0_CFG75
dir_path=/home/nzilberstein/red_diff_stable/RED-diff_stable/particle_guidance/stable_diffusion/generated_images_SDS/steps_999_w_100.0_second_False_seed_6_dino_True_coeff_0.0_lr_0.01
python eval_clip_score_particle_SDS.py --gpu $gpu --dir_path $dir_path --csv_file $csv_file

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/10.0
# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/ambient_space/10.0
# python eval_clip_score_particle_SDS.py --gpu $gpu --dir_path $dir_path --csv_file $csv_file

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/10.0
# # dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/ambient_space/10.0
# python eval_clip_score_particle_SDS.py --gpu $gpu --dir_path $dir_path --csv_file $csv_file

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/20.0
# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/n_part/20.0
# python eval_clip_score_particle_SDS.py --gpu $gpu --dir_path $dir_path --csv_file $csv_file

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/30.0
# python eval_clip_score_particle_SDS.py --gpu $gpu --dir_path $dir_path --csv_file $csv_file

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/40.0
# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/40.0/1000iter
# python eval_clip_score_particle_SDS.py --gpu $gpu --dir_path $dir_path --csv_file $csv_file

# dir_path=/home/nzilberstein/red_diff_stable/prolific_dreamer2d/generated_images/ancestral
# python eval_clip_score_particle_SDS.py --gpu 0 --dir_path $dir_path --csv_file $csv_file
