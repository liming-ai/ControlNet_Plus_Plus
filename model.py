from __future__ import annotations

import gc

import numpy as np
import PIL.Image
import torch
from controlnet_aux.util import HWC3
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from cv_utils import resize_image
from preprocessor import Preprocessor
from settings import MAX_IMAGE_RESOLUTION, MAX_NUM_IMAGES

CONTROLNET_MODEL_IDS = {
    "Canny": "checkpoints/canny/controlnet",

    "softedge": "checkpoints/hed/controlnet",

    "segmentation": "checkpoints/seg/controlnet",

    "depth": "checkpoints/depth/controlnet",

    "lineart": "checkpoints/lineart/controlnet",
}


def download_all_controlnet_weights() -> None:
    for model_id in CONTROLNET_MODEL_IDS.values():
        ControlNetModel.from_pretrained(model_id)


class Model:
    def __init__(self, base_model_id: str = "runwayml/stable-diffusion-v1-5", task_name: str = "Canny"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model_id = ""
        self.task_name = ""
        self.pipe = self.load_pipe(base_model_id, task_name)
        self.preprocessor = Preprocessor()

    def load_pipe(self, base_model_id: str, task_name) -> DiffusionPipeline:
        if (
            base_model_id == self.base_model_id
            and task_name == self.task_name
            and hasattr(self, "pipe")
            and self.pipe is not None
        ):
            return self.pipe
        model_id = CONTROLNET_MODEL_IDS[task_name]
        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id, safety_checker=None, controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if self.device.type == "cuda":
            pipe.disable_xformers_memory_efficient_attention()
        pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.base_model_id = base_model_id
        self.task_name = task_name
        return pipe

    def set_base_model(self, base_model_id: str) -> str:
        if not base_model_id or base_model_id == self.base_model_id:
            return self.base_model_id
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        try:
            self.pipe = self.load_pipe(base_model_id, self.task_name)
        except Exception:
            self.pipe = self.load_pipe(self.base_model_id, self.task_name)
        return self.base_model_id

    def load_controlnet_weight(self, task_name: str) -> None:
        if task_name == self.task_name:
            return
        if self.pipe is not None and hasattr(self.pipe, "controlnet"):
            del self.pipe.controlnet
        torch.cuda.empty_cache()
        gc.collect()
        model_id = CONTROLNET_MODEL_IDS[task_name]
        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
        controlnet.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe.controlnet = controlnet
        self.task_name = task_name

    def get_prompt(self, prompt: str, additional_prompt: str) -> str:
        if not prompt:
            prompt = additional_prompt
        else:
            prompt = f"{prompt}, {additional_prompt}"
        return prompt

    @torch.autocast("cuda")
    def run_pipe(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: PIL.Image.Image,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        generator = torch.Generator().manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            generator=generator,
            image=control_image,
        ).images

    @torch.inference_mode()
    def process_canny(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        low_threshold: int,
        high_threshold: int,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        self.preprocessor.load("Canny")
        control_image = self.preprocessor(
            image=image, low_threshold=low_threshold, high_threshold=high_threshold, detect_resolution=image_resolution
        )

        self.load_controlnet_weight("Canny")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        conditions_of_generated_imgs = [
            self.preprocessor(
                image=x, low_threshold=low_threshold, high_threshold=high_threshold, detect_resolution=image_resolution
            ) for x in results
        ]
        return [control_image] * num_images + results + conditions_of_generated_imgs

    @torch.inference_mode()
    def process_mlsd(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        value_threshold: float,
        distance_threshold: float,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        self.preprocessor.load("MLSD")
        control_image = self.preprocessor(
            image=image,
            image_resolution=image_resolution,
            detect_resolution=preprocess_resolution,
            thr_v=value_threshold,
            thr_d=distance_threshold,
        )
        self.load_controlnet_weight("MLSD")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_scribble(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name == "HED":
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                scribble=False,
            )
        elif preprocessor_name == "PidiNet":
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                safe=False,
            )
        self.load_controlnet_weight("scribble")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_scribble_interactive(
        self,
        image_and_mask: dict[str, np.ndarray],
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        if image_and_mask is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        image = image_and_mask["mask"]
        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        control_image = PIL.Image.fromarray(image)

        self.load_controlnet_weight("scribble")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_softedge(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name in ["HED", "HED safe"]:
            safe = "safe" in preprocessor_name
            self.preprocessor.load("HED")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                scribble=safe,
            )
        elif preprocessor_name in ["PidiNet", "PidiNet safe"]:
            safe = "safe" in preprocessor_name
            self.preprocessor.load("PidiNet")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                safe=safe,
            )
        else:
            raise ValueError
        self.load_controlnet_weight("softedge")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        conditions_of_generated_imgs = [
            self.preprocessor(
                image=x,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                scribble=safe,
            ) for x in results
        ]
        return [control_image] * num_images + results + conditions_of_generated_imgs

    @torch.inference_mode()
    def process_openpose(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load("Openpose")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                hand_and_face=True,
            )
        self.load_controlnet_weight("Openpose")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_segmentation(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        self.load_controlnet_weight("segmentation")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        self.preprocessor.load('UPerNet')
        conditions_of_generated_imgs = [
            self.preprocessor(
                image=np.array(x),
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            ) for x in results
        ]
        return [control_image] * num_images + results + conditions_of_generated_imgs

    @torch.inference_mode()
    def process_depth(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        self.load_controlnet_weight("depth")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        conditions_of_generated_imgs = [
            self.preprocessor(
                image=x,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            ) for x in results
        ]
        return [control_image] * num_images + results + conditions_of_generated_imgs

    @torch.inference_mode()
    def process_normal(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load("NormalBae")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        self.load_controlnet_weight("NormalBae")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_lineart(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name in ["None", "None (anime)"]:
            image = 255 - HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name in ["Lineart", "Lineart coarse"]:
            coarse = "coarse" in preprocessor_name
            self.preprocessor.load("Lineart")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                coarse=coarse,
            )
        elif preprocessor_name == "Lineart (anime)":
            self.preprocessor.load("LineartAnime")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        # NOTE: We still use the general lineart model
        if "anime" in preprocessor_name:
            self.load_controlnet_weight("lineart_anime")
        else:
            self.load_controlnet_weight("lineart")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        self.preprocessor.load("Lineart")
        conditions_of_generated_imgs = [
            self.preprocessor(
                image=x,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            ) for x in results
        ]

        control_image = PIL.Image.fromarray((255 - np.array(control_image)).astype(np.uint8))
        conditions_of_generated_imgs = [PIL.Image.fromarray((255 - np.array(x)).astype(np.uint8)) for x in conditions_of_generated_imgs]

        return [control_image] * num_images + results + conditions_of_generated_imgs

    @torch.inference_mode()
    def process_shuffle(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
            )
        self.load_controlnet_weight("shuffle")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_ip2p(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        control_image = PIL.Image.fromarray(image)
        self.load_controlnet_weight("ip2p")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results
