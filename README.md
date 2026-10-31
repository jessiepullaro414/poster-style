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

## `illustrate.py` -- img2img + ControlNet, redraws the image

Actually repaints the photo as flat shapes/cel shading using a ControlNet (lineart
or canny) so the subject's outline is preserved while the surface style is
regenerated. Much closer to a hand-drawn poster illustration than `posterize.py`.
Needs a GPU for reasonable speed. Two models, picked with `--model`:

- **sd15** (default) -- ~10s/image at 768px on an 8GB RTX 3070 Ti. lineart or canny.
- **sdxl** -- sharper detail and truer color, closer to the hand-drawn look, but
  doesn't fit an 8GB card at its native 1024px -- runs under CPU-offload at
  **~10-12 minutes/image** on the same GPU. canny only (no SDXL lineart
  checkpoint wired up yet). Worth it for a final render, not for iterating on
  a prompt.

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
# optional, only needed for --remove-bg:
./.venv/Scripts/python.exe -m pip install rembg onnxruntime
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

First run of a given `--model` downloads its weights into `./models`, cached for
every run after: ~6 GB for sd15 (base + ControlNet lineart + annotator), ~9 GB
for sdxl (base + ControlNet canny + the fp16-fix VAE).

```bash
./.venv/Scripts/python.exe illustrate.py car.jpg -o car_art.png \
  --extra-prompt "description of the specific subject helps a lot"

./.venv/Scripts/python.exe illustrate.py car.jpg -o car_art_sdxl.png --model sdxl \
  --extra-prompt "description of the specific subject helps a lot"
```

Key knobs (all in the module docstring / `--help`):

- `--strength` (default 0.95) -- img2img denoising. **Below ~0.8 these models
  barely restyle a real photo** -- they just denoise back toward the original,
  so don't be shy pushing this up. Much above ~0.95 it starts drifting off the
  ControlNet outline.
- `--control-scale` (default 0.65) -- how hard the lineart/canny edges are
  enforced. Raise it if fine details (grille, mirror, lights) are melting away;
  lower it if the background is coming out dark/forest-like (a high scale drags
  the source photo's shadowed regions straight into the linework) or the output
  still looks too photographic.
- `--extra-prompt` -- describe the actual subject (colors, materials, setting).
  Doing this well matters more than any of the numeric knobs. Both models'
  CLIP text encoders truncate at 77 tokens (~60 words including the base
  prompt) -- keep it to the details that actually matter, most important first.
- `--seed` -- fix it to compare parameter changes on the same noise draw.
- `--model sdxl` -- see the speed/quality tradeoff above. Everything else
  (`--strength`, `--control-scale`, `--extra-prompt`, `--seed`) works the same
  way; `--control`, `--max-size`, `--steps`, and `--guidance` get sdxl-specific
  defaults automatically unless you pass them explicitly.
- `--remove-bg` -- cut the subject out (rembg, local/free) and flatten onto
  white before generating. Gets rid of a busy real-world background *and*
  its cast shadow, which ControlNet otherwise carries straight into the
  output as if they were part of the subject. Needs `rembg`/`onnxruntime`
  (see setup above); first use downloads a ~1GB segmentation model.
- `--pastel` -- post-process color grade toward a softer palette (reuses
  `posterize.py`'s `grade()`). Generated colors tend to run more saturated
  than a poster wants; cheaper and more predictable than re-rolling seeds to
  chase color taste.
- `--style-image` (IP-Adapter, condition directly on a reference image
  instead of describing it in words) is wired up but **currently broken** in
  this venv -- a diffusers 0.40.0 / transformers 5.x incompatibility with no
  fix available as of diffusers' latest release. Fails fast with an
  explanation rather than burning GPU time. See the comment above the check
  in `illustrate.py` if revisiting this later.

### What actually worked, end to end

For the reference photo this was built against, the combination that landed
closest to a hand-drawn poster: `--remove-bg` (kills the real cast shadow and
busy lawn/driveway background) + SD1.5 (flatter than SDXL, which kept pulling
back toward photoreal) + `--pastel` (SD1.5's raw color choices ran punchier
than a poster wants) + a specific `--extra-prompt` describing the subject's
actual colors and details + a seed that happened to land on the right wheel
color (`--seed 23` here -- there's no shortcut for this part, roll a few).

## Running on the garage PC (CPU only)

The Beelink at the workbench (Ryzen 7 7735U, 8 cores, Radeon iGPU -- no CUDA)
runs `illustrate.py` on its CPU at `C:\Users\Garage\poster-style`. Slow, but
it's always on, so the laptop doesn't have to be.

Setup there: Python 3.13 (`winget install Python.Python.3.13 --scope user`),
then a `.venv` with CPU-only wheels (no 2.5 GB CUDA build needed):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.9.1+cpu torchvision==0.24.1+cpu diffusers==0.40.0 transformers==5.17.0 tokenizers==0.23.2 accelerate==1.15.0 safetensors controlnet_aux==0.0.10 opencv-python numpy pillow rembg==2.0.85 onnxruntime==1.30.0
```

Models were copied over the LAN from the main PC rather than re-downloaded:
SD 1.5, the lineart ControlNet and the annotators into `models/hub`, and the
rembg models (`bria-rmbg`, `isnet-general-use`) into `~\.rembg\models`. SDXL
isn't there -- on this CPU it would take well over an hour per image.

Run through `garage_run.ps1`, which starts the worker at below-normal priority
(so the wall-screen dashboard keeps the CPU when it needs it) and prints wall
time + peak memory. From the main PC:

```bash
tar -cf - photo.jpg | ssh garage@192.168.1.110 'tar -xf - -C C:\Users\Garage\poster-style'
ssh garage@192.168.1.110 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Garage\poster-style\garage_run.ps1 -In C:\Users\Garage\poster-style\photo.jpg -Out C:\Users\Garage\poster-style\photo_art.png -Prompt "subject description"'
ssh garage@192.168.1.110 'tar -cf - -C C:\Users\Garage\poster-style photo_art.png' | tar -xf -
```

Measured (reference Alfa photo, SD 1.5, `--remove-bg --pastel`, 26 steps):
**377 s per image** (11.5 s/step + ~75 s loading/cutout), **9.5 GB peak
private memory**. The same seed gives a different image than on the GPU (CPU
and CUDA generate different noise) but is reproducible run to run here.

Memory, measured phase by phase: background removal was the peak (BRIA
RMBG-2.0 hit 10.5 GB, more than the whole diffusion pipeline). Fixed by doing
every cutout before the pipeline loads, through one shared session, and turning
off ONNX Runtime's CPU memory arena (10.45 -> 7.77 GB for the cutout step).
`--bg-model isnet-general-use` drops that step to 1.7 GB / 6 s, with slightly
soft edges where the tires meet the ground. Attention slicing and VAE tiling
made no difference on CPU, so they stay GPU-only.
