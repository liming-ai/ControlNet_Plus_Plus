export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_canny"
export REWARDMODEL_DIR="canny"
export OUTPUT_DIR="work_dirs/reward_model/MultiGen20M_train/reward_canny_res512_bs256_lr1e-5_warmup100_scale-1_iter10k_fp16_train0-1k_reward0-200_denormalized-img_magnitude-mse-loss_threshold-0.1-0.2"


accelerate launch --config_file "train/config.yml" \
 --main_process_port=23156 train/reward_control.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="canny" \
 --dataset_name="limingcv/MultiGen-20M_train" \
 --caption_column="text" \
 --conditioning_image_column="canny" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=4 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=8 \
 --max_train_steps=10000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=0 \
 --checkpointing_steps=500 \
 --grad_scale=1.0 \
 --use_ema \
 --validation_steps=100 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200