import os
import re
import gc
import argparse
import random
import torch
import safetensors.torch
from PIL import Image
from typing import Optional, Union, Literal, Dict, cast

from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

class Inference:
    def __init__(self, model_weights, sdxl=False, disable_torch_compile=False):
        if sdxl:
            self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                model_weights,
                vae=self.vae,
                use_safetensors=True,
                torch_dtype=torch.float16,
            ).to('cuda')
        else:
            self.vae = AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors")
            self.pipe = StableDiffusionPipeline.from_single_file(
                model_weights,
                vae=self.vae,
                use_safetensors=True,
                torch_dtype=torch.float16,
            ).to('cuda')
        if not disable_torch_compile:
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
    @staticmethod
    def process_prompt_args(prompt: str, sdxl=False):
        prompt_args = prompt.split(" --")
        prompt = prompt_args[0]
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        num_inference_steps = 30
        width = height = 1024
        guidance_scale = 12
        seed = None
        images_per_prompt = 1
        
        for parg in prompt_args:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                width = int(m.group(1))
                continue
            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                height = int(m.group(1))
                continue
            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                seed = int(m.group(1))
                continue
            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                num_inference_steps = max(1, min(1000, int(m.group(1))))
                continue
            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                guidance_scale = float(m.group(1))
                continue
            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                negative_prompt = m.group(1)
                continue
            m = re.match(r"t (\d+)", parg, re.IGNORECASE)
            if m:
                images_per_prompt = int(m.group(1))
                continue
                
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        height = max(64, height - height % 8)
        width = max(64, width - width % 8)
        
        return prompt, negative_prompt, num_inference_steps, width, height, guidance_scale, seed, images_per_prompt
        
    def get_scheduler(self, name="Euler a"):
        # Get scheduler
        match name:
            case "DPM++ 2M":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config
                )

            case "DPM++ 2M Karras":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "DPM++ 2M SDE":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++"
                )

            case "DPM++ 2M SDE Karras":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++",
                )

            case "DPM++ SDE":
                return DPMSolverSinglestepScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "DPM++ SDE Karras":
                return DPMSolverSinglestepScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "DPM2":
                return KDPM2DiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "DPM2 Karras":
                return KDPM2DiscreteScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "Euler":
                return EulerDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "Euler a":
                return EulerAncestralDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "Heun":
                return HeunDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "LMS":
                return LMSDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "LMS Karras":
                return LMSDiscreteScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "DDIM":
                return DDIMScheduler.from_config(self.pipe.scheduler.config)

            case "DEISMultistep":
                return DEISMultistepScheduler.from_config(self.pipe.scheduler.config)

            case "UniPCMultistep":
                return UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
                
        return EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)  # default to "Euler a"

    def validate(self, prompt_string, image_output, sdxl=False):
        self.pipe.scheduler = self.get_scheduler()
        
        prompt, negative_prompt, num_inference_steps, width, height, guidance_scale, seed, num_images_per_prompt = self.process_prompt_args(prompt_string, sdxl=sdxl)

        generator = torch.Generator().manual_seed(seed)
        
        print("\nInference Parameters:")
        print(f"  - Prompt: {prompt}")
        print(f"  - Negative Prompt: {negative_prompt}")
        print(f"  - Image Width: {width}")
        print(f"  - Image Height: {height}")
        print(f"  - Num Inference Steps: {num_inference_steps}")
        print(f"  - Guidance Scale: {guidance_scale}")
        print(f"  - Images per Prompt: {num_images_per_prompt}\n")

        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        ).images

        return image

if prompt:
        infer = Inference(output_path, sdxl=sdxl, disable_torch_compile=disable_torch_compile)
        images = infer.validate(prompt, image_output, sdxl=sdxl)
        free_memory()
    
        if image_output:
            basename = os.path.splitext(os.path.basename(image_output))[0]
            dirpath = os.path.dirname(image_output)
            
            if isinstance(images, list):
                for idx, img in enumerate(images):
                    img.save(os.path.join(dirpath, f'{basename}_{idx}.png'))
            else:
                images.save(image_output)
    
        return images

def load_checkpoint(checkpoint_path):
    """
    Load a PyTorch checkpoint file (.ckpt).

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        dict: Loaded checkpoint (model weights, optimizer state, etc.).
    """
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def main():
    args = parse_arguments()

    checkpoint_merger(
        interpolation=args.interpolation,
        multiplier=args.multiplier,
        half=args.half,
        discard_weights=args.discard_weights,
        validate=args.validate,
        prompt=args.prompt,
        image_output=args.image_output,
        sampler=args.sampler,
        sdxl=args.sdxl,
        disable_torch_compile=args.disable_torch_compile,

    # Load the checkpoint
    checkpoint = load_checkpoint(args.checkpoint_path)

    # Perform inference using the loaded checkpoint
    # Add your inference code here

    print("Inference completed!")

if __name__ == "__main__":
    main()
