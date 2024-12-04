# dir_path=/home/nzilberstein/Inverse/_exp/output_rip/RED_diff_nonaug
# dest_folder=/home/nzilberstein/Inverse/_exp/output_rip/RED_diff_nonaug/FID

# python eval_FID.py --dir_path $dir_path --dest_folder $dest_folder

# python -m pytorch_fid /home/nzilberstein/Inverse/_exp/input/FFHQ_512 $dest_folder

# dir_path=/home/nzilberstein/repository/constrained_sampling/_exp/output/rsd_stable_repulsion_gamma50/inp_large_box
# dest_folder=/home/nzilberstein/repository/constrained_sampling/_exp/output/rsd_stable/FID

# python eval_FID.py --dir_path $dir_path --dest_folder $dest_folder

# python -m pytorch_fid /home/nzilberstein/Inverse/_exp/input/FFHQ_512 $dest_folder

# dir_path=/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/output/PSLD/output_mb_large/samples
# dest_folder=/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/output/PSLD/output_mb_large/FID

# python eval_FID.py --dir_path $dir_path --dest_folder $dest_folder

# python -m pytorch_fid /home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/FFHQ_512 $dest_folder



# python -m pytorch_fid /home/nzilberstein/repository/constrained_sampling/_exp/output/rsd_stablenon_aug/FID /home/nzilberstein/repository/constrained_sampling/_exp/input/FFHQ_512

dir_path=/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/output/PSLD/output_bip_large_imagenet/samples
dest_folder=/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/output/PSLD/output_bip_large_imagenet/samples/FID
# dir_path=/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/output/rsd_stable/inp_free_imagenet
# dest_folder=/home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/output/rsd_stable/inp_free_imagenet/FID
python eval_FID.py --dir_path $dir_path --dest_folder $dest_folder

python -m pytorch_fid /home/nicolas/Repulsive-score-distillation-RSD-/constrained_sampling/_exp/input/IMAGENET_512 $dest_folder
