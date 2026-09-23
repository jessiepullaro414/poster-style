#!/usr/bin/env python3
"""Turn a photograph into a flat, poster-style illustration.

The pipeline flattens regions, collapses the palette, grades the colour toward a
muted/matte look, lays in soft outlines, and finishes with a faint grain -- the
combination that makes a photo read as a printed illustration rather than a
snapshot.

    python posterize.py car.jpg -o car_poster.png
    python posterize.py *.jpg --outdir out --suffix _poster

Every stage has a knob; run with --help for the full list.  Needs numpy and
opencv-python.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def fit_width(img, max_width):
    """Downscale so the long edge is <= max_width (mean-shift is O(pixels))."""
    if max_width <= 0:
        return img
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA)


def flatten_regions(img, sp, sr, bilateral_passes, d, sigma_color, sigma_space):
    """Collapse gradients into flat areas while keeping edges crisp."""
    out = img
    if sp > 0 and sr > 0:
        out = cv2.pyrMeanShiftFiltering(out, sp, sr)
    for _ in range(bilateral_passes):
        out = cv2.bilateralFilter(out, d, sigma_color, sigma_space)
    return out


def quantize(img, k, seed):
    """k-means colour quantisation in Lab space (perceptual grouping)."""
    if k <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    samples = lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(
        samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    quant = centers.astype(np.uint8)[labels.flatten()].reshape(lab.shape)
    return cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)


def posterize_levels(img, levels):
    """Optional extra per-channel level reduction."""
    if levels <= 0 or levels >= 256:
        return img
    step = 255.0 / (levels - 1)
    return (np.round(img / step) * step).clip(0, 255).astype(np.uint8)


def grade(img, saturation, warmth, lift, gamma, contrast):
    """Muted, slightly warm, lifted-black 'matte print' grade."""
    f = img.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 1)
    f = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # BGR: index 2 is red, index 0 is blue
    f[..., 2] = np.clip(f[..., 2] * (1.0 + warmth), 0, 1)
    f[..., 0] = np.clip(f[..., 0] * (1.0 - warmth), 0, 1)

    f = lift + (1.0 - lift) * f                       # raise the floor
    f = np.clip((f - 0.5) * contrast + 0.5, 0, 1)     # pivot contrast on mid grey
    f = np.power(np.clip(f, 0, 1), gamma)

    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def add_outlines(img, strength, thickness):
    """Soft dark edges where the flattened image changes colour."""
    if strength <= 0:
        return img
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    flat = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    edge = 255 - flat
    if thickness > 1:
        edge = cv2.dilate(edge, np.ones((thickness, thickness), np.uint8))
    edge = cv2.GaussianBlur(edge, (0, 0), 0.8)

    alpha = (edge.astype(np.float32) / 255.0 * strength)[..., None]
    out = img.astype(np.float32) * (1.0 - alpha)      # darken toward the line
    return np.clip(out, 0, 255).astype(np.uint8)


def add_grain(img, amount, seed):
    """Faint luminance noise so flat areas aren't dead flat."""
    if amount <= 0:
        return img
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, img.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)[..., None] * (amount * 255.0)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def stylize(img, a):
    img = fit_width(img, a.max_width)
    img = flatten_regions(
        img, a.meanshift_sp, a.meanshift_sr,
        a.bilateral_passes, a.bilateral_d, a.bilateral_sigma_color, a.bilateral_sigma_space,
    )
    img = quantize(img, a.colors, a.seed)
    img = cv2.medianBlur(img, 3)                       # kill speckle on region borders
    img = posterize_levels(img, a.posterize)
    img = grade(img, a.saturation, a.warmth, a.lift, a.gamma, a.contrast)
    img = add_outlines(img, a.outline, a.outline_thickness)
    img = add_grain(img, a.grain, a.seed)
    return img


def resolve_outputs(inputs, args):
    if args.output:
        if len(inputs) != 1:
            sys.exit("-o/--output takes exactly one input; use --outdir for batches")
        return [args.output]
    outs = []
    for path in inputs:
        stem, ext = os.path.splitext(os.path.basename(path))
        out_ext = ext if ext.lower() in (".png", ".jpg", ".jpeg", ".webp") else ".png"
        outdir = args.outdir or os.path.dirname(path) or "."
        outs.append(os.path.join(outdir, f"{stem}{args.suffix}{out_ext}"))
    return outs


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("inputs", nargs="+", help="input image(s); globs are expanded")
    p.add_argument("-o", "--output", help="output path (single input only)")
    p.add_argument("--outdir", help="directory for batch output (default: alongside input)")
    p.add_argument("--suffix", default="_poster", help="filename suffix for batch output")

    g = p.add_argument_group("size")
    g.add_argument("--max-width", type=int, default=1600,
                   help="downscale so width <= this before processing (0 = keep)")

    g = p.add_argument_group("region flattening")
    g.add_argument("--meanshift-sp", type=float, default=18.0, help="spatial radius (0 = off)")
    g.add_argument("--meanshift-sr", type=float, default=28.0, help="colour radius")
    g.add_argument("--bilateral-passes", type=int, default=2)
    g.add_argument("--bilateral-d", type=int, default=9)
    g.add_argument("--bilateral-sigma-color", type=float, default=90.0)
    g.add_argument("--bilateral-sigma-space", type=float, default=9.0)

    g = p.add_argument_group("palette")
    g.add_argument("--colors", type=int, default=16, help="k-means colours (0 = off)")
    g.add_argument("--posterize", type=int, default=0,
                   help="extra per-channel levels, e.g. 24 (0 = off)")

    g = p.add_argument_group("colour grade")
    g.add_argument("--saturation", type=float, default=0.82, help="<1 mutes, >1 boosts")
    g.add_argument("--warmth", type=float, default=0.05, help="red up / blue down")
    g.add_argument("--lift", type=float, default=0.06, help="raise the black floor (matte)")
    g.add_argument("--gamma", type=float, default=0.94)
    g.add_argument("--contrast", type=float, default=0.92, help="<1 softens")

    g = p.add_argument_group("finishing")
    g.add_argument("--outline", type=float, default=0.28, help="edge darkness 0..1 (0 = off)")
    g.add_argument("--outline-thickness", type=int, default=2)
    g.add_argument("--grain", type=float, default=0.015, help="noise amount 0..1 (0 = off)")
    g.add_argument("--seed", type=int, default=0)

    return p


def main():
    args = build_parser().parse_args()

    inputs = []
    for pattern in args.inputs:
        hits = glob.glob(pattern)
        inputs.extend(sorted(hits) if hits else [pattern])

    outputs = resolve_outputs(inputs, args)

    for src, dst in zip(inputs, outputs):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip (unreadable): {src}", file=sys.stderr)
            continue
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        result = stylize(img, args)
        if not cv2.imwrite(dst, result):
            print(f"failed to write: {dst}", file=sys.stderr)
            continue
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
