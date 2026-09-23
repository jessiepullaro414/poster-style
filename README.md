# poster-style

Two ways to turn a photo into a flat-illustration "poster" image. Both are local,
free, and need no account or API key.

## `posterize.py` -- classical filter, instant, no ML

Bilateral/mean-shift flattening + k-means palette + colour grade + soft outlines +
grain. Runs in ~1-2s on a CPU, any Python with `opencv-python`/`numpy`. Produces a
cutout/poster look but keeps real photo detail (reflections stay as photo noise,
not redrawn shapes) -- see the module docstring for tuning knobs.

```bash
python posterize.py car.jpg -o car_poster.png
```

## `illustrate.py` -- Stable Diffusion img2img + ControlNet, redraws the image

Actually repaints the photo as flat shapes/cel shading using SD 1.5 + a ControlNet
(lineart by default) so the subject's outline is preserved while the surface style
is regenerated. Much closer to a hand-drawn poster illustration. Needs a GPU for
reasonable speed (tested on an 8 GB RTX 3070 Ti: ~10s/image at 768px, 28 steps).

### One-time setup

Requires a **Python 3.13** interpreter -- as of writing, PyPI/pytorch.org don't
ship CUDA wheels for 3.13's successor yet, only a CPU-only one, which is much too
slow. If you only have a newer Python installed, grab a standalone 3.13 with
[uv](https://github.com/astral-sh/uv) (no admin rights needed):

```bash
pip install uv
uv python install 3.13
```

Then build the venv (adjust the path uv printed for your 3.13 install):

```bash
<path-to-python-3.13>/python.exe -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
# CUDA build -- match the torch/torchvision versions to each other (see pytorch.org
# for current cu12x tag + compatible torchvision version if these have moved on):
./.venv/Scripts/python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
./.venv/Scripts/python.exe -m pip install diffusers transformers accelerate safetensors numpy pillow opencv-python controlnet_aux
```

**Gotcha:** installing the second line (diffusers etc.) with no `--index-url` can
let pip silently upgrade `torch` back to the default PyPI **CPU-only** build if any
package's dependency resolution nudges the version. After that install, always
verify:

```bash
./.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it says `+cpu` / `False`, reinstall pinned with `--force-reinstall --no-deps`
from the cu128 index for both `torch` and `torchvision` (versions must match each
other -- check https://download.pytorch.org/whl/cu128/torch/ and
.../torchvision/ for a pair built for the same cp3xx tag).

### Usage

First run downloads ~6 GB of weights (SD 1.5 + ControlNet lineart + annotator)
into `./models`, cached for every run after.

```bash
./.venv/Scripts/python.exe illustrate.py car.jpg -o car_art.png \
  --extra-prompt "description of the specific subject helps a lot"
```

Key knobs (all in the module docstring / `--help`):

- `--strength` (default 0.9) -- img2img denoising. **Below ~0.8, SD1.5 barely
  restyles a real photo** -- it just denoises back toward the original, so don't
  be shy pushing this up. Above ~0.95 it starts drifting off the ControlNet
  outline.
- `--control-scale` (default 0.75) -- how hard the lineart/canny edges are
  enforced. Raise it if fine details (grille, mirror, lights) are melting away;
  lower it if the output still looks too photographic.
- `--extra-prompt` -- describe the actual subject (colors, materials, setting).
  Doing this well matters more than any of the numeric knobs.
- `--seed` -- fix it to compare parameter changes on the same noise draw.
