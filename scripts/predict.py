"""Segment your own images from a handful of example masks.

This is the USE path, as opposed to `sota_final.py`, which is the benchmark path. The benchmark
harness speaks in support pools, test splits and seeds because it has to make few-shot variance
honest; none of that applies when you simply have some annotated images and some unannotated ones.

    python scripts/predict.py --support supp/ --images raw/ --out masks/

`--support` is a directory of example pairs: an image and its mask sharing a stem, with the mask
distinguished by a suffix (default `_mask`), e.g.

    supp/cells_01.png        supp/cells_01_mask.png
    supp/cells_02.tif        supp/cells_02_mask.tif

A mask may be binary (anything > 0 is foreground) or a per-instance label map; either way it is read
as foreground versus background, because the method predicts SEMANTIC FOREGROUND -- a per-pixel
foreground map, not separated instances. Around eight examples is the operating point the paper
reports; one already works, and past sixteen the curve is flat on most datasets.

Everything in `--images` is then segmented and written to `--out` as a PNG mask of the same size,
plus, with `--prob`, the raw probability map as a 32-bit TIFF for your own thresholding.

Requires a CUDA GPU and access to the gated DINOv3 weights (`huggingface-cli login`).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


def _read(path):
    """Read an image as HxW or HxWxC uint8/uint16, without a colour conversion of its own.

    The colour channel the method feeds its bank is chosen from the support masks at fit time, so
    converting to grayscale here would throw away exactly the signal that choice depends on.
    """
    import skimage.io
    try:
        return skimage.io.imread(path)
    except ValueError as e:
        # Microscopy TIFFs are very often LZW- or JPEG-compressed, and tifffile needs imagecodecs to
        # decode those. Its own error names the codec but not the fix, which sends the user hunting.
        if "imagecodecs" in str(e):
            raise SystemExit(f"{path}: this TIFF uses a compression tifffile cannot decode on its "
                             f"own. Run 'pip install imagecodecs' and try again. ({e})")
        raise


def _pairs(d, suffix):
    """(image, mask) paths in `d`, matched by stem. Fails loud on an unmatched image.

    A silently skipped pair is the worst outcome here: the run still succeeds, the segmenter just
    fits on fewer examples than the user believes it did, and the quality drop looks like the
    method being weak rather than like an input that never arrived.
    """
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
    masks = {os.path.splitext(f)[0]: f for f in files if os.path.splitext(f)[0].endswith(suffix)}
    out = []
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem.endswith(suffix):
            continue
        if stem + suffix not in masks:
            raise SystemExit(f"{os.path.join(d, f)}: no mask '{stem}{suffix}.*' beside it. Every "
                             f"support image needs one; rename or move the file, or pass a "
                             f"different --mask-suffix.")
        out.append((os.path.join(d, f), os.path.join(d, masks[stem + suffix])))
    if not out:
        raise SystemExit(f"{d}: no support pairs found (looked for '*{suffix}.*' beside each image)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--support", required=True, help="directory of example image/mask pairs")
    p.add_argument("--images", required=True, help="directory of images to segment")
    p.add_argument("--out", required=True, help="directory to write masks into")
    p.add_argument("--mask-suffix", default="_mask", help="support mask stem suffix (default: _mask)")
    p.add_argument("--res", type=int, default=672, help="encoder resolution (default: 672, as reported)")
    p.add_argument("--cache", default=None,
                   help="feature cache directory (default: <out>/.feature_cache). Encoding is the "
                        "expensive step, so caching makes a second run over the same images nearly "
                        "free. NEVER point two concurrent runs at one cache directory.")
    p.add_argument("--prob", action="store_true", help="also write the float probability map as TIFF")
    p.add_argument("--method", default="head_fusion_best_cgate_film_nobank",
                   help="method string (default: the configuration reported in the paper)")
    args = p.parse_args()

    import skimage.io
    import torch
    from al_testbed import make_backend
    from active_segmenter.config import RunConfig, EncoderConfig
    from active_segmenter.encoder.cached import CachedEncoder
    from active_segmenter.segment.base import LabeledExample

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: no CUDA device found. This will work but be very slow.", flush=True)

    sup_pairs = _pairs(args.support, args.mask_suffix)
    queries = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMG_EXT))
    if not queries:
        raise SystemExit(f"{args.images}: no images to segment")
    os.makedirs(args.out, exist_ok=True)
    cache = args.cache or os.path.join(args.out, ".feature_cache")

    cfg = RunConfig(device=dev, cache_dir=cache,
                    encoder=EncoderConfig(resolution=args.res))
    enc = CachedEncoder(cfg, dev, cache)
    be = make_backend(args.method, cfg, dev, enc=enc, support_k=len(sup_pairs))

    print(f"fitting on {len(sup_pairs)} example(s) ...", flush=True)
    support = []
    for ip, mp in sup_pairs:
        im, m = _read(ip), _read(mp)
        if m.ndim == 3:                       # a mask saved as RGB: collapse, any channel set = fg
            m = m.max(-1)
        if m.shape[:2] != im.shape[:2]:
            raise SystemExit(f"{mp}: mask is {m.shape[:2]} but {os.path.basename(ip)} is "
                             f"{im.shape[:2]}; they must match pixel for pixel")
        support.append(LabeledExample(im, enc.extract(im), (np.asarray(m) > 0).astype(int)))
    be.fit(support)                            # this is where the method configures ITSELF

    for i, f in enumerate(queries, 1):
        im = _read(os.path.join(args.images, f))
        prob = be.foreground_prob(im, enc.extract(im))
        stem = os.path.splitext(f)[0]
        skimage.io.imsave(os.path.join(args.out, f"{stem}_mask.png"),
                          ((prob > 0.5).astype(np.uint8) * 255), check_contrast=False)
        if args.prob:
            skimage.io.imsave(os.path.join(args.out, f"{stem}_prob.tif"),
                              prob.astype(np.float32), check_contrast=False)
        print(f"  [{i}/{len(queries)}] {f} -> {stem}_mask.png "
              f"({100 * float((prob > 0.5).mean()):.1f}% foreground)", flush=True)

    print(f"done: {len(queries)} mask(s) in {args.out}", flush=True)


if __name__ == "__main__":
    main()
