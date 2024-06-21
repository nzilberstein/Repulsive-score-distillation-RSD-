input="/home/nzilberstein/repository/unconstrained_sampling/prolific_dreamer2d/datacoco/subset.csv"
counter=0
steps=500
gamma=0
gpu=6

while IFS= read -r line
do
        counter=$((counter+1))

        # Skip the first line
        if [ "$counter" -eq 1 ]; then
         continue
        fi

        echo counter: $counter
        # Use awk to extract the last field from each line
        last_field=$(echo "$line" | awk -F ',' '{print $NF}')
        
        echo "$last_field"

        python prolific_dreamer2d.py \
        --num_steps $steps --log_steps 100 \
        --seed 6 --lr 0.03 --phi_lr 0.0001 --use_t_phi true \
        --model_path 'stabilityai/stable-diffusion-2-1-base' \
        --loss_weight_type '1m_alphas_cumprod' --t_schedule 'random' \
        --generation_mode 'vsd' \
        --phi_model 'lora' --lora_scale 1. --lora_vprediction false \
        --prompt "$last_field" \
        --height 512 --width 512 --batch_size 4 --guidance_scale 7.5 \
        --particle_num_vsd 4 --particle_num_phi 4 \
        --log_progress false --save_x0 false --save_phi_model true --coeff $gamma --gpu $gpu --dino --index_folder $counter
        

done < "$input"