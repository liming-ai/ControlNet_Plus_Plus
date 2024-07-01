# ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback (ECCV 2024)

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.07987-b31b1b.svg)](https://arxiv.org/abs/2404.07987)&nbsp;
[![huggingface demo](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-ControlNet++-yellow)](https://huggingface.co/spaces/limingcv/ControlNet-Plus-Plus)&nbsp;

</div>
<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2404.07987">ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback</a>
</p>

<p align="center">
<img src="images/github_imgs/teaser.png" width=95%>
<p>

<br>

## 📢 We will release all the training code and organized data this Friday.

## 🕹️ Try and Play with ControlNet++

We provide a [demo website](https://huggingface.co/spaces/limingcv/ControlNet-Plus-Plus) for you to play with our ControlNet++ models and generate images interactively. For local running, please run the following command:
```bash
git clone https://github.com/liming-ai/ControlNet_Plus_Plus.git
pip3 install -r requirements.txt
```

Please download the model weights and put them into each subset of `checkpoints`:
|   model    |HF weights🤗                                                                        |
|:----------:|:------------------------------------------------------------------------------------|
|  LineArt   | [model](https://huggingface.co/spaces/limingcv/ControlNet-Plus-Plus/resolve/main/checkpoints/lineart/controlnet/diffusion_pytorch_model.bin) |
|  Depth   |  [model](https://huggingface.co/spaces/limingcv/ControlNet-Plus-Plus/resolve/main/checkpoints/depth/controlnet/diffusion_pytorch_model.safetensors) |
|  Segmentation   | [model](https://huggingface.co/spaces/limingcv/ControlNet-Plus-Plus/resolve/main/checkpoints/seg/controlnet/diffusion_pytorch_model.safetensors) |
|  Hed (SoftEdge)   | [model](https://huggingface.co/spaces/limingcv/ControlNet-Plus-Plus/resolve/main/checkpoints/hed/controlnet/diffusion_pytorch_model.bin) |
| Canny | [model](https://huggingface.co/spaces/limingcv/ControlNet-Plus-Plus/resolve/main/checkpoints/canny/controlnet/diffusion_pytorch_model.safetensors) |

And then run:
```bash
python3 app.py
```


## What's new for ControlNet++?

### ✨ Cycle Consistency for Conditional Generation

We model image-based controllable generation as an image translation task from input conditional controls $c_v$ to output generated images $x'_0$. If we translate images from one domain to the other (condition $c_v$ → generated image $x'_0$ ), and back again (generated image $x'_0$ → condition $c_v'$ ) we should arrive where we started ($c_v$ = $c_v'$). Hence, we can directly optimize the cycle consistency loss for better controllability.

<p align="center">
<img src="https://liming-ai.github.io/ControlNet_Plus_Plus/static/images/cycle_consistency.png" width=95%>
<p>

### ✨ Directly Optimization for Controllability:
**(a)** Existing methods achieve implicit controllability by introducing imagebased conditional control $c_v$ into the denoising process of diffusion models, with the guidance of latent-space denoising loss. **(b)** We utilize discriminative reward models $D$ to explicitly optimize the controllability of $G$ via pixel-level cycle consistency loss.
<p align="center">
<img src="https://liming-ai.github.io/ControlNet_Plus_Plus/static/images/comparison.png" width=95%>
<p>

### ✨ Efficient Reward Strategy:
**(a)** Pipeline of default reward fine-tuning strategy. Reward fine-tuning requires sampling all the way to the full image. Such a method needs to keep all gradients for each timestep and the memory required is unbearable by current GPUs. **(b)** We add a small noise ($t ≤ t_{thre}$) to disturb the consistency between input images and conditions, then the single-step denoised image can be directly used for efficient reward fine-tuning.
<p align="center">
<img src="https://liming-ai.github.io/ControlNet_Plus_Plus/static/images/efficient_reward.png" width=95%>
<p>


### 🔥 Better Controllability Than Existing Methods (Qualitative Results):


<p align="center">
<img src="images/github_imgs/vis_comparison.png" width=95%>
<p>



### 🔥 Better Controllability Than Existing Methods (Quantitative Results):

<p align="center">
<img src="https://liming-ai.github.io/ControlNet_Plus_Plus/static/images/results.png" width=95%>
<p>

#### For a deep dive into our analyses, discussions, and evaluations, check out our [paper](https://arxiv.org/abs/2404.07987).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@article{li2024controlnet,
    author  = {Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, Chen Chen},
    title   = {ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback},
    journal = {arXiv preprint arXiv:2404.07987},
    year    = {2024},
}
```
