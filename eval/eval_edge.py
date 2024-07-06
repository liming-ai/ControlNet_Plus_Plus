import os
import json
import torch
import argparse
import numpy as np
import torch.multiprocessing as mp

from PIL import Image
from einops import rearrange
from collections import defaultdict
from torchmetrics.classification import BinaryF1Score
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

def eval_canny(pool_args):

    anno, anno_dir, img_dir, device, ssim, psnr, fid, f1, task = pool_args
    gt_condition_path = os.path.join(anno_dir, anno)
    image_paths = [os.path.join(img_dir, f'group_{i}', anno.replace('.png', '_edge.png')) for i in range(4)]

    gt_condition = torch.from_numpy(np.array(Image.open(gt_condition_path).convert('L'))).to(device)
    gt_condition = rearrange(gt_condition, 'h w -> 1 1 h w').repeat(4, 1, 1, 1)
    pred_condition = [torch.from_numpy(np.array(Image.open(path).convert('L'))).to(device) for path in image_paths]
    pred_condition = torch.stack(pred_condition)
    pred_condition = rearrange(pred_condition, 'b h w -> b 1 h w')

    # Assuming ssim and psnr instances are created and moved to the correct device beforehand
    ssim_score = ssim((pred_condition/255.0).clip(0,1), (gt_condition/255.0).clip(0,1))
    psnr_score = psnr((pred_condition/255.0).clip(0,1), (gt_condition/255.0).clip(0,1))

    gt_condition[gt_condition == 255] = 1
    pred_condition[pred_condition == 255] = 1

    f1_score = f1(pred_condition.flatten(), gt_condition.flatten())

    # When two conditions are indentical, the psnr_score is inf, so we set a maximum value for it
    if torch.isinf(psnr_score):
        psnr_score = torch.tensor(100.0, device=device)

    return {
        'f1': f1_score.item(),
        'ssim': ssim_score.item(),
        'psnr': psnr_score.item(),
    }


def eval_hed_lineart(pool_args):
    anno, anno_dir, img_dir, device, ssim, psnr, fid, f1, task = pool_args
    gt_condition_path = os.path.join(anno_dir, anno)
    image_paths = [os.path.join(img_dir, f'group_{i}', anno.replace('.png', '_lineart.png')) for i in range(4)]

    gt_condition = torch.from_numpy(np.array(Image.open(gt_condition_path).convert('L'))).to(device)
    gt_condition = rearrange(gt_condition, 'h w -> 1 1 h w').repeat(4, 1, 1, 1)
    pred_condition = [torch.from_numpy(np.array(Image.open(path).convert('L'))).to(device) for path in image_paths]
    pred_condition = torch.stack(pred_condition)
    pred_condition = rearrange(pred_condition, 'b h w -> b 1 h w')

    if task == 'lineart':
        pred_condition = 1 - pred_condition

    # Assuming ssim and psnr instances are created and moved to the correct device beforehand
    ssim_score = ssim((pred_condition/255.0).clip(0,1), (gt_condition/255.0).clip(0,1))
    psnr_score = psnr((pred_condition/255.0).clip(0,1), (gt_condition/255.0).clip(0,1))

    if torch.isinf(psnr_score):
        psnr_score = torch.tensor(100.0, device=device)

    return {
        'ssim': ssim_score.item(),
        'psnr': psnr_score.item(),
    }

def main(root_dir, num_processes, task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anno_dir = os.path.join(root_dir, 'annotations')
    img_dir = os.path.join(root_dir, 'images')

    ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    f1 = BinaryF1Score().to(device)
    fid = FID(normalize=True).to(device)

    anno_files = os.listdir(anno_dir)
    # Include the device in the arguments passed to each worker function
    pool_args = [(anno, anno_dir, img_dir, device, ssim, psnr, fid, f1, task) for anno in anno_files]

    mp.set_start_method('spawn', force=True)
    # Remove the initargs as worker_init_fn cannot accept additional arguments
    with mp.Pool(processes=num_processes) as pool:
        if task == 'canny':
            results = pool.map(eval_canny, pool_args)
        elif task in ['hed', 'lineart']:
            results = pool.map(eval_hed_lineart, pool_args)
        else:
            raise ValueError(f'Invalid task: {task}')

    sums = defaultdict(int)
    counts = defaultdict(int)
    for result in results:
        for key, value in result.items():
            sums[key] += value
            counts[key] += 1
    metrics = {key: sums[key] / counts[key] for key in sums}
    metrics['Number of Samples'] = len(anno_files)
    print(metrics)

    save_dir = args.root_dir.replace('eval_dirs', 'eval')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(metrics, f)
    print(f"Evaluation results are saved to: {os.path.join(save_dir, 'results.json')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate edge tasks")
    parser.add_argument('--task', type=str, default='canny', help='Edge tasks, including canny, hed, and lineart')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the evaluation data')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use')

    args = parser.parse_args()

    # You should pass the 'task' argument to the main function as well
    main(args.root_dir, args.num_processes, args.task)