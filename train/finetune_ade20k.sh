# "lllyasviel/control_v11p_sd15_seg" is trained on both ADE20K and COCOStuff
# Here we finetune on seperate dataset to get better results
# And then we reward fine-tuning the fine-tuned models
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_seg"
export OUTPUT_DIR="work_dirs/finetune/Captioned_ADE20K/ft_controlnet_sd15_seg_res512_bs256_lr1e-5_warmup100_iter5k_fp16"


accelerate launch --config_file "train/config.yml" \
 --main_process_port=23156 train/train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="limingcv/Captioned_ADE20K" \
 --caption_column="prompt" \
 --conditioning_image_column="control_seg" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=4 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=32 \
 --max_train_steps=5000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=200 \
 --checkpointing_steps=500 \
 --validation_steps=100 \