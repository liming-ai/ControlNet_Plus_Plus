import os
import argparse
from PIL import Image
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F

def compute_per_pixel_mse(root_dir):
    anno_dir = os.path.join(root_dir, 'annotations')
    img_dir = os.path.join(root_dir, 'images')

    accs = []
    bar = tqdm(range(len(os.listdir(anno_dir))))
    for anno in os.listdir(anno_dir):
        label_path = os.path.join(anno_dir, anno)
        image_paths = [os.path.join(
            img_dir, f'group_{i}', anno.replace('.png', '_depth.png')
        ) for i in range(4)]

        label = torch.from_numpy(np.array(Image.open(label_path).convert('L')))
        preds = [torch.from_numpy(np.array(Image.open(path).convert('L'))) for path in image_paths]
        per_pixel_mse = [torch.sqrt(F.mse_loss(pred.float(), label.float())) for pred in preds]
        accs.append(per_pixel_mse)
        bar.update(1)

    mean_mse = np.array(accs).mean()
    print("Mean Per-Pixel RMSE:", mean_mse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-pixel RMSE for images.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing annotations and images")
    args = parser.parse_args()

    compute_per_pixel_mse(args.root_dir)