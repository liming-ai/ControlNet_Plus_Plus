# Path to the controlnet weight (can be huggingface or local path)
# export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_seg"  # Eval ControlNet
# export CONTROLNET_DIR="checkpoints/ade20k_reward-model-FCN-R101-d8/checkpoint-5000/controlnet"  # Eval our ControlNet++
export CONTROLNET_DIR="checkpoints/ade20k_reward-model-UperNet-R50/checkpoint-5000/controlnet"    # Eval our ControlNet++

# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=8
# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=20

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --main_process_port=23456 --num_processes=$NUM_GPUS eval/eval.py --task_name='seg' --dataset_name='limingcv/Captioned_ADE20K' --dataset_split='validation' --condition_column='control_seg' --prompt_column='prompt' --label_column='seg_map' --model_path=${CONTROLNET_DIR} --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
export DATA_DIR="work_dirs/eval_dirs/Captioned_ADE20K/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# Evaluation with mmseg api
mim test mmseg mmlab/mmseg/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py \
    --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth \
    --gpus 8 \
    --launcher pytorch \
    --cfg-options test_dataloader.dataset.datasets.0.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.1.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.2.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.3.data_root="${DATA_DIR}" \
                  work_dir="${DATA_DIR}"