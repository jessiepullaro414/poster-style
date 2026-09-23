#!/usr/bin/env python3
"""Repaint a photo as a flat editorial illustration -- fully local, no API, no cost.

Stable Diffusion 1.5 img2img with a ControlNet (lineart or canny) so the subject's
geometry survives while the surface style is redrawn into flat shapes and cel
shading.  The first run downloads ~6 GB of open weights from HuggingFace into
./models (override with --hf-home or the HF_HOME env var); no account needed.

    python illustrate.py photo.jpg -o out.png
    python illustrate.py photo.jpg -o out.png --strength 0.7 --control lineart --seed 3
    python illustrate.py *.jpg --outdir out --suffix _art

Run inside poster-style/.venv (see the project setup notes).
"""

import argparse
import glob
import os
import sys

# Keep the multi-GB model cache next to this script unless the user says otherwise.
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, "models"))

import numpy as np
import torch
from PIL import Image


BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNETS = {
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "canny": "lllyasviel/control_v11p_sd15_canny",
}

DEFAULT_PROMPT = (
    "flat editorial illustration, minimalist vector art style, smooth cel shading, "
    "clean flat color shapes, muted pastel palette, soft ambient lighting, matte finish, "
    "subtle canvas texture, modern poster art, crisp linework"
)
DEFAULT_NEGATIVE = (
    "photograph, photorealistic, 3d render, hdr, film grain, noise, jpeg artifacts, "
    "harsh specular highlights, oversaturated, text, watermark, signature, blurry, "
    "deformed, wrong proportions, extra wheels, extra lights"
)


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def load_resized(path, max_size):
    """Open RGB and scale so the long edge == max_size, both edges multiples of 8."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_size / max(w, h)
    nw, nh = (max(64, round(w * scale)), max(64, round(h * scale)))
    nw, nh = (nw - nw % 8, nh - nh % 8)
    return img.resize((nw, nh), Image.LANCZOS)


def make_control_image(img, kind):
    if kind == "canny":
        import cv2

        arr = cv2.Canny(np.array(img), 100, 200)
        arr = np.stack([arr] * 3, axis=-1)
        control = Image.fromarray(arr)
    else:
        from controlnet_aux import LineartDetector

        detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        control = detector(img, coarse=False)

    # controlnet_aux/cv2 snap to their own resolution grid (multiples of 64) --
    # force an exact match to the img2img latent size or the U-Net add fails.
    if control.size != img.size:
        control = control.resize(img.size, Image.LANCZOS)
    return control


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def build_pipeline(control_kind):
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionImg2ImgPipeline,
        UniPCMultistepScheduler,
    )

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32

    if control_kind == "none":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL, torch_dtype=dtype, safety_checker=None
        )
    else:
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNETS[control_kind], torch_dtype=dtype
        )
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            BASE_MODEL, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
        )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if cuda:
        pipe.to("cuda")
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
    else:
        print("no CUDA -- running on CPU, expect minutes per image", file=sys.stderr)
    pipe.set_progress_bar_config(leave=False)
    return pipe


def run_one(pipe, img, control_kind, args, seed):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    kw = dict(
        prompt=args.prompt,
        negative_prompt=args.negative,
        image=img,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    if control_kind != "none":
        kw["control_image"] = make_control_image(img, control_kind)
        kw["controlnet_conditioning_scale"] = args.control_scale
    return pipe(**kw).images[0]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def resolve_outputs(inputs, args):
    if args.output:
        if len(inputs) != 1:
            sys.exit("-o/--output takes exactly one input; use --outdir for batches")
        return [args.output]
    outs = []
    for path in inputs:
        stem, _ = os.path.splitext(os.path.basename(path))
        outdir = args.outdir or os.path.dirname(path) or "."
        outs.append(os.path.join(outdir, f"{stem}{args.suffix}.png"))
    return outs


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("inputs", nargs="+", help="input image(s); globs are expanded")
    p.add_argument("-o", "--output", help="output path (single input only)")
    p.add_argument("--outdir", help="directory for batch output")
    p.add_argument("--suffix", default="_art", help="filename suffix for batch output")

    p.add_argument("--control", choices=["lineart", "canny", "none"], default="lineart",
                   help="how to lock geometry (default: lineart)")
    p.add_argument("--control-scale", type=float, default=0.75,
                   help="ControlNet strength 0..1.5 (higher = more faithful outline)")
    p.add_argument("--strength", type=float, default=0.9,
                   help="img2img denoising 0..1 (higher = more restyled, less faithful). "
                        "Below ~0.8 SD1.5 barely restyles a photo -- it just denoises "
                        "back toward the original.")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=8.0)
    p.add_argument("--max-size", type=int, default=768, help="long edge in pixels")
    p.add_argument("--seed", type=int, default=-1, help="-1 = random per image")

    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative", default=DEFAULT_NEGATIVE)
    p.add_argument("--extra-prompt", default="",
                   help="appended to --prompt, e.g. subject description")

    p.add_argument("--hf-home", help="override model cache dir (default: ./models)")
    return p


def main():
    args = build_parser().parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.extra_prompt:
        args.prompt = f"{args.prompt}, {args.extra_prompt}"

    inputs = []
    for pattern in args.inputs:
        hits = glob.glob(pattern)
        inputs.extend(sorted(hits) if hits else [pattern])
    outputs = resolve_outputs(inputs, args)

    print(f"loading pipeline (control={args.control}) ...", flush=True)
    pipe = build_pipeline(args.control)

    for src, dst in zip(inputs, outputs):
        if not os.path.exists(src):
            print(f"skip (missing): {src}", file=sys.stderr)
            continue
        seed = torch.seed() % (2**31) if args.seed < 0 else args.seed
        img = load_resized(src, args.max_size)
        print(f"{src}  {img.size}  seed={seed}", flush=True)
        result = run_one(pipe, img, args.control, args, seed)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        result.save(dst)
        print(f"  -> {dst}", flush=True)


if __name__ == "__main__":
    main()
