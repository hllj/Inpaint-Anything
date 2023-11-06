import os
import sys
import glob
import argparse
import torch
import numpy as np
import PIL.Image as Image
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline
from diffusers import UniPCMultistepScheduler

from utils.mask_processing import crop_for_filling_pre, crop_for_filling_post
from utils.crop_for_replacing import recover_size, resize_and_pad
from utils import load_img_to_array, save_array_to_img


def fill_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        device="cuda"
):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
    img_crop, mask_crop = crop_for_filling_pre(img, mask)
    img_crop_filled = pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_crop),
        mask_image=Image.fromarray(mask_crop)
    ).images[0]
    img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
    return img_filled


def replace_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        prompts,
        neg_promt,
        step: int = 50,
        device="cuda"
):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
    height, width, _ = img.shape

    imgs_gen = pipe(
        prompt=prompts,
        negative_prompt=[neg_promt] * len(prompts),
        image=Image.fromarray(img_padded),
        mask_image=Image.fromarray(255 - mask_padded),
        num_inference_steps=step,
        strength=0.75,
        num_images_per_prompt=2
    ).images
    
    results = []
    for img_gen in imgs_gen:
        img_resized, mask_resized = recover_size(
            np.array(img_gen), mask_padded, (height, width), padding_factors)
        mask_resized = np.expand_dims(mask_resized, -1) / 255
        img_resized = img_resized * (1-mask_resized) + img * mask_resized
        img_resized = img_resized.astype(np.uint8)
        results.append(img_resized)
    
    return results


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--text_prompt", type=str, required=True,
        help="Text prompt",
    )
    parser.add_argument(
        "--input_mask_glob", type=str, required=True,
        help="Glob to input masks",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Specify seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms for reproducibility.",
    )

if __name__ == "__main__":
    """Example usage:
    python lama_inpaint.py \
        --input_img FA_demo/FA1_dog.png \
        --input_mask_glob "results/FA1_dog/mask*.png" \
        --text_prompt "a teddy bear on a bench" \
        --output_dir results 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    img_stem = Path(args.input_img).stem
    mask_ps = sorted(glob.glob(args.input_mask_glob))
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_img_to_array(args.input_img)
    for mask_p in mask_ps:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        mask = load_img_to_array(mask_p)
        img_filled_p = out_dir / f"filled_with_{Path(mask_p).name}"
        img_filled = fill_img_with_sd(
            img, mask, args.text_prompt, device=device)
        save_array_to_img(img_filled, img_filled_p)