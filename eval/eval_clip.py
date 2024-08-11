# pip3 install torchmetrics
import torch
import argparse

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.multimodal.clip_score import CLIPScore

parser = argparse.ArgumentParser(description="Evaluate CLIP-Score")
parser.add_argument('--generated_image_dir', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True, default="limingcv/MultiGen-20M_canny_eval")
args = parser.parse_args()

dataset = load_dataset(args.dataset, cache_dir='data/huggingface_datasets', split='validation')
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()

bar = tqdm(range(len(dataset)), desc=f"Evaluating {args.dataset}")

rewards = []
for idx in range(len(dataset)):
    data = dataset[idx]
    prompt = data["text"]

    image_paths = [f'{args.generated_image_dir}/group_{i}/{idx}.png' for i in range(4)]
    images = [Image.open(x).convert('RGB') for x in image_paths]
    images = [pil_to_tensor(x).cuda() for x in images]
    metric.update(torch.stack(images), [prompt]*4)

    bar.update(1)

print(metric.score / metric.n_samples)