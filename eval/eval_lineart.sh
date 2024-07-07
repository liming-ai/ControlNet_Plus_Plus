# Path to the controlnet weight (can be huggingface or local path)
# export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_lineart"  # Eval ControlNet
export CONTROLNET_DIR="checkpoints/lineart/controlnet"  # Eval our ControlNet++
# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=8
# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=20

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --main_process_port=12116 --num_processes=8 eval/eval.py --task_name='lineart' --dataset_name='limingcv/MultiGen-20M_canny_eval' --dataset_split='validation' --condition_column='image' --prompt_column='text' --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_canny_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# Run the evaluation code
python3 eval/eval_edge.py --task lineart --root_dir ${DATA_DIR} --num_processes ${NUM_GPUS}