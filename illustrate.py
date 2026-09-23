#!/usr/bin/env python3
"""Repaint a photo as a flat editorial illustration -- fully local, no API, no cost.

img2img with a ControlNet (lineart or canny) so the subject's geometry survives
while the surface style is redrawn into flat shapes and cel shading. Two models:

- sd15 (default): fast (~10s/image on an 8GB GPU), lineart or canny, ~5GB VRAM.
- sdxl: sharper detail and colour, closer to a hand-drawn poster, canny only
  here. Doesn't fit an 8GB card at its native 1024px even with attention
  slicing, so it runs under CPU-offload -- ~10-12 minutes/image on an RTX
  3070 Ti. Drop --max-size (e.g. 768) to trade resolution for speed.

First run of a given model downloads its weights from HuggingFace into
./models (override with --hf-home or the HF_HOME env var); no account needed.

    python illustrate.py photo.jpg -o out.png
    python illustrate.py photo.jpg -o out.png --model sdxl --seed 3
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


BASE_MODELS = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}
CONTROLNETS = {
    "sd15": {
        "lineart": "lllyasviel/control_v11p_sd15_lineart",
        "canny": "lllyasviel/control_v11p_sd15_canny",
    },
    "sdxl": {
        # diffusers' official SDXL canny checkpoint; there's no equally
        # well-supported single-purpose SDXL lineart checkpoint yet, so
        # --control lineart is sd15-only for now.
        "canny": "diffusers/controlnet-canny-sdxl-1.0",
    },
}
SDXL_VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"  # avoids NaN/black frames in fp16
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHTS = {
    "sd15": ("models", "ip-adapter_sd15.bin"),
    "sdxl": ("sdxl_models", "ip-adapter_sdxl.bin"),
}

DEFAULT_PROMPT = (
    "flat editorial illustration, minimalist vector art style, smooth cel shading, "
    "solid flat color shapes with no reflections, muted pastel palette, soft even "
    "daylight, matte finish, modern poster art, crisp linework"
)
DEFAULT_NEGATIVE = (
    "photograph, photorealistic, 3d render, hdr, film grain, noise, jpeg artifacts, "
    "glossy, reflective, specular highlight, shiny paint reflection, glare, chrome "
    "shine, dark background, black foliage, night, dim, dense dark forest, harsh "
    "shadow, high contrast, oversaturated, text, watermark, signature, blurry, "
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


def load_style_image(path):
    """Open a reference image for IP-Adapter -- flatten transparency onto white
    first (a background-removed PNG's RGB under a transparent pixel is
    undefined and can come back black, which would poison the CLIP embedding)."""
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def _smooth_for_edges(img):
    """Bilateral-smooth before edge detection. Canny/lineart on a raw photo
    pick up the *boundary* of a specular highlight/reflection as a hard edge
    indistinguishable from a real panel line -- ControlNet then locks that
    blotchy shape into the output no matter what the prompt says, which is
    why "no reflections" alone doesn't remove them. This erases soft
    photographic gradients (reflections, paint sheen) while preserving
    genuine sharp edges (panel gaps, grille, window frames), so only real
    structure reaches the edge detector."""
    import cv2

    arr = np.array(img)
    arr = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
    arr = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(arr)


def make_control_image(img, kind):
    smoothed = _smooth_for_edges(img)
    if kind == "canny":
        import cv2

        arr = cv2.Canny(np.array(smoothed), 100, 200)
        arr = np.stack([arr] * 3, axis=-1)
        control = Image.fromarray(arr)
    else:
        from controlnet_aux import LineartDetector

        detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        control = detector(smoothed, coarse=False)

    # controlnet_aux/cv2 snap to their own resolution grid (multiples of 64) --
    # force an exact match to the img2img latent size or the U-Net add fails.
    if control.size != img.size:
        control = control.resize(img.size, Image.LANCZOS)
    return control


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def build_pipeline(model, control_kind, use_style=False, style_strength=0.6):
    import diffusers

    if control_kind != "none" and control_kind not in CONTROLNETS[model]:
        sys.exit(
            f"--control {control_kind} isn't wired up for --model {model} "
            f"(available: {', '.join(CONTROLNETS[model])}, or 'none')"
        )

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    base = BASE_MODELS[model]
    variant = "fp16" if (cuda and model == "sdxl") else None

    if model == "sdxl":
        vae = diffusers.AutoencoderKL.from_pretrained(SDXL_VAE_FIX, torch_dtype=dtype)
        if control_kind == "none":
            pipe = diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained(
                base, vae=vae, torch_dtype=dtype, variant=variant, use_safetensors=True
            )
        else:
            controlnet = diffusers.ControlNetModel.from_pretrained(
                CONTROLNETS[model][control_kind], torch_dtype=dtype
            )
            pipe = diffusers.StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                base, controlnet=controlnet, vae=vae, torch_dtype=dtype,
                variant=variant, use_safetensors=True,
            )
    else:
        if control_kind == "none":
            pipe = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
                base, torch_dtype=dtype, safety_checker=None
            )
        else:
            controlnet = diffusers.ControlNetModel.from_pretrained(
                CONTROLNETS[model][control_kind], torch_dtype=dtype
            )
            pipe = diffusers.StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                base, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
            )

    pipe.scheduler = diffusers.UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if use_style:
        # Must happen before .to(cuda)/enable_model_cpu_offload -- offload's
        # device hooks are registered against the components present at that
        # point, and the IP-Adapter's image encoder needs to be one of them.
        subfolder, weight_name = IP_ADAPTER_WEIGHTS[model]
        pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=subfolder, weight_name=weight_name)
        pipe.set_ip_adapter_scale(style_strength)

    if cuda:
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if model == "sdxl":
            # SDXL + a ControlNet doesn't fit an 8GB card at 1024px. Calling
            # plain .to("cuda") doesn't even raise OOM on Windows -- WDDM
            # silently spills the overflow into slow shared system memory,
            # which measured *slower* than explicit offload (~16 vs ~12
            # min/image, same output, tested back to back). Offload it is.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        print("no CUDA -- running on CPU, expect minutes per image", file=sys.stderr)

    pipe.set_progress_bar_config(leave=False)
    return pipe


def run_one(pipe, img, control_kind, args, seed, style_image=None):
    # With enable_model_cpu_offload (SDXL) pipe.device can read back "cpu" even
    # though generation runs on the GPU -- generate noise on the real compute
    # device instead of trusting that attribute.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)
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
    if style_image is not None:
        kw["ip_adapter_image"] = style_image
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

    p.add_argument("--model", choices=["sd15", "sdxl"], default="sd15",
                   help="sd15: fast, ~5GB VRAM, lineart or canny. "
                        "sdxl: sharper/more detail, ~1024px native, canny only, "
                        "needs CPU-offload on an 8GB card (default: sd15)")
    p.add_argument("--control", choices=["lineart", "canny", "none"], default=None,
                   help="how to lock geometry (default: lineart for sd15, canny for sdxl)")
    p.add_argument("--control-scale", type=float, default=0.65,
                   help="ControlNet strength 0..1.5 (higher = more faithful outline, but "
                        "also drags dark/shadowed regions from the source photo's "
                        "linework straight into the output -- lower it if the "
                        "background comes out too dark/forest-like)")
    p.add_argument("--strength", type=float, default=0.95,
                   help="img2img denoising 0..1 (higher = more restyled, less faithful). "
                        "Below ~0.8 these models barely restyle a photo -- they just "
                        "denoise back toward the original.")
    p.add_argument("--steps", type=int, default=None,
                   help="default: 28 for sd15, 32 for sdxl")
    p.add_argument("--guidance", type=float, default=None,
                   help="default: 7.5 for sd15, 6.0 for sdxl (SDXL wants lower CFG)")
    p.add_argument("--max-size", type=int, default=None,
                   help="long edge in pixels (default: 768 for sd15, 1024 for sdxl)")

    p.add_argument("--style-image",
                   help="reference image (IP-Adapter) whose look gets blended in on "
                        "top of the text prompt -- much stronger pull toward matching "
                        "a specific target image than prompt wording alone. A "
                        "background-removed / isolated-subject crop works best since "
                        "its background won't compete with the prompt's.")
    p.add_argument("--style-strength", type=float, default=0.6,
                   help="IP-Adapter weight 0..1 -- how hard --style-image pulls "
                        "vs. the text prompt (default 0.6)")
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

    # per-model defaults for the knobs left unset on the command line
    if args.control is None:
        args.control = "canny" if args.model == "sdxl" else "lineart"
    if args.max_size is None:
        args.max_size = 1024 if args.model == "sdxl" else 768
    if args.steps is None:
        args.steps = 32 if args.model == "sdxl" else 28
    if args.guidance is None:
        args.guidance = 6.0 if args.model == "sdxl" else 7.5

    inputs = []
    for pattern in args.inputs:
        hits = glob.glob(pattern)
        inputs.extend(sorted(hits) if hits else [pattern])
    outputs = resolve_outputs(inputs, args)

    style_image = load_style_image(args.style_image) if args.style_image else None
    if style_image is not None:
        # Confirmed broken in this venv (diffusers 0.40.0 + transformers 5.17.0):
        # load_ip_adapter() patches pipe.unet's attention processors, but its
        # image-embedding path hits `encoder_hidden_states.shape` on what comes
        # back as a plain tuple -- reproduces identically with or without
        # ControlNet, so it's not a ControlNet interaction. diffusers 0.40.0 is
        # still the latest release (no fix yet), and transformers can't be
        # downgraded to the 4.x line diffusers' IP-Adapter code targets without
        # cascading into incompatible tokenizers/huggingface-hub pins in this
        # stack. Fail fast here instead of burning 10+ GPU-minutes on a crash --
        # delete this guard once a compatible diffusers/transformers pair exists.
        sys.exit(
            "--style-image (IP-Adapter) doesn't work in this venv right now: "
            "diffusers 0.40.0's IP-Adapter code is incompatible with "
            "transformers 5.x here (AttributeError: 'tuple' object has no "
            "attribute 'shape'), and there's no newer diffusers release to "
            "pull. See the comment above this check in illustrate.py."
        )

    print(f"loading pipeline (model={args.model}, control={args.control}, "
          f"style={'yes' if style_image else 'no'}) ...", flush=True)
    pipe = build_pipeline(args.model, args.control,
                           use_style=style_image is not None,
                           style_strength=args.style_strength)

    for src, dst in zip(inputs, outputs):
        if not os.path.exists(src):
            print(f"skip (missing): {src}", file=sys.stderr)
            continue
        seed = torch.seed() % (2**31) if args.seed < 0 else args.seed
        img = load_resized(src, args.max_size)
        print(f"{src}  {img.size}  seed={seed}", flush=True)
        result = run_one(pipe, img, args.control, args, seed, style_image=style_image)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        result.save(dst)
        print(f"  -> {dst}", flush=True)


if __name__ == "__main__":
    main()
