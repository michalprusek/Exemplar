#!/usr/bin/env python
"""BiomedParse baseline on the AutoSeg panel — SELF-CONTAINED (does not import al_testbed).

Benchmarks **BiomedParse** (Zhao et al., *Nature Methods* 2024/2025, repo microsoft/BiomedParse)
on our few-shot panel at the MATCHED test protocol so its per-dataset foreground numbers sit
directly beside ours.

WHAT THIS MEASURES — be honest about the paradigm
-------------------------------------------------
BiomedParse is a **TEXT-PROMPTED biomedical segmentation FOUNDATION model** (trained on 6M+
image/mask/text triples across 9 modalities). It is **K-INDEPENDENT**: it segments an object from
a natural-language prompt, NOT from K support (image,mask) pairs. So the 8 support masks are
IGNORED; there is ONE pass over the fixed test split (no seed loop → seeds=[0], one prediction
per image). We consume the exact same ``load_dataset(spec, pool, test, seed=0)`` FIXED test split
and score with the SAME ``active_segmenter.eval.scoring.score_prediction`` metric fn as ours, so
the comparison is protocol-matched. Label it honestly in any write-up:
**BiomedParse = text-prompted foundation model (K-independent), our method = 8-shot.**

VOCABULARY / FAIRNESS (critical — read before trusting any number)
------------------------------------------------------------------
BiomedParse's 9 modalities are radiology/pathology/ophthalmology-centric — there is **NO light-
microscopy or fluorescence-microscopy modality**, and its fine target vocabulary (target_dist.json)
is:
  Pathology : connective tissue cells | inflammatory cells | neoplastic cells | epithelial cells
  Endoscopy : polyp | neoplastic polyp | non-neoplastic polyp
  Fundus    : optic cup | optic disc            <-- NO retinal-vessel target!
  Dermoscopy: lesion | melanoma
  OCT       : edema
  (+ CT / MRI / X-Ray / Ultrasound organ targets)
Consequence for OUR panel (per-dataset prompt is the CLOSEST in-vocabulary/in-modality target):
  * monuseg      H&E pathology nuclei  -> "neoplastic cells"  IN-MODALITY (partial: cells != nuclei)
  * kvasir       endoscopy polyp       -> "polyp"             IN-DOMAIN   (clean match; the one fair +)
  * drive / hrf  fundus retinal vessel -> "vessel"            BORDERLINE  (vessel is a coarse
                 BIOMED_CLASS but NOT a Fundus fine-target; fundus vessels are effectively OOD)
  * spheroidj / dsb2018 / ctc_u373  light/fluor microscopy cells -> OOD (no microscopy modality)
  * microtubules thin filaments        -> N/A (no filament/neurite/microtubule target anywhere)
Expect LOW numbers on the OOD rows — that is an HONEST finding (foundation model out of its trained
modality distribution), not a setup bug. The ``vocab`` column in the summary flags each row.

METRIC (matched to ours; foreground only)
-----------------------------------------
BiomedParse emits a per-prompt foreground probability map (no instance decoder), so we score the
FOREGROUND with the dataset-appropriate metric, overriding instance-AP -> fg-IoU on the blob/
instance datasets (the task's directive):
  * blobs / instances (spheroidj, dsb2018, monuseg, ctc_u373, kvasir) -> fg-IoU vs (label_map>0)
  * vessels / filaments (drive, hrf, microtubules)                    -> clDice
Per-image scores are written in the SAME json contract as ``scripts/sota_final.py`` so the ``stats``
stage aggregates BiomedParse next to ours:
  {"method","dataset","metric","test_per_seed":N,"seeds":[0],"per_image":[float,...]}

SETUP (tulen) — see results/BASELINE-SETUP.md for the full recipe
-----------------------------------------------------------------
BiomedParse's 2D text-prompted model lives on the repo's **``main``** branch (the default HEAD has
moved to ``v2`` = the CVPR2025 *3D* challenge refactor, which does NOT contain the 2D
``modeling/`` + ``utilities/`` + ``configs/biomedparse_inference.yaml``). Weights =
``biomedparse_v1.pt`` on the **GATED** HF repo ``microsoft/BiomedParse`` (must accept the terms of
use on the HF web page first — a token alone is not enough). Fresh Python-3.9 env with the custom
``detectron2-xyz`` fork + mpi4py + deepspeed per ``assets/requirements/requirements.txt``.

Run (once weights + env are ready):
  export BIOMEDPARSE_SRC=/disk1/prusek/incontext/BiomedParse       # with `main` checked out
  export PYTHONPATH=/disk1/prusek/active-segmenter
  export HF_HOME=/disk1/prusek/.cache/huggingface
  CUDA_VISIBLE_DEVICES=1 <biomedparse_env>/bin/python scripts/biomedparse_bench.py \
      --datasets all --score_dir results/scores_basekscale/biomedparse
  # smoke (1 image / dataset):
  ... scripts/biomedparse_bench.py --smoke --datasets monuseg,kvasir,hrf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

# --- make active_segmenter.eval.{registry,scoring} importable (numpy/skimage only) ----------
_ASG_ROOT = os.environ.get("ASG_ROOT", "/disk1/prusek/active-segmenter")
if _ASG_ROOT not in sys.path:
    sys.path.insert(0, _ASG_ROOT)

from active_segmenter.eval.registry import PANEL, load_dataset  # noqa: E402
from active_segmenter.eval.scoring import primary_key, score_prediction  # noqa: E402

# Per-dataset (closest-in-vocabulary text prompt, scoring-metric override, vocab-status). The
# metric override follows the task directive: fg-IoU on blob/instance rows, clDice on vessel/
# filament rows. ``vocab`` documents fairness: in_domain > in_modality > borderline > ood > n/a.
#   prompt=None  -> honestly N/A: no in-vocabulary target exists; the dataset is SKIPPED.
PROMPTS = {
    # blob / instance datasets -> fg-IoU
    "monuseg":      ("neoplastic cells", "iou", "in_modality"),   # H&E pathology (cells != nuclei)
    "kvasir":       ("polyp",            "iou", "in_domain"),     # endoscopy polyp — clean match
    "spheroidj":    ("neoplastic cells", "iou", "ood"),          # light microscopy — no LM modality
    "dsb2018":      ("neoplastic cells", "iou", "ood"),          # fluorescence nuclei — OOD
    "ctc_u373":     ("neoplastic cells", "iou", "ood"),          # phase-contrast cells — OOD
    # vessel / filament datasets -> clDice
    "drive":        ("vessel",           "cldice", "borderline"),  # coarse class, not a fundus target
    "hrf":          ("vessel",           "cldice", "borderline"),
    "microtubules": (None,               "cldice", "n/a"),         # no filament/neurite target
}
# Report order (the task's panel + the kvasir in-domain anchor).
PANEL_ORDER = ["spheroidj", "dsb2018", "monuseg", "ctc_u373", "drive", "hrf", "microtubules", "kvasir"]


# ------------------------------------------------------------------------------------------
def to_rgb_pil(img):
    """Return a PIL RGB image BiomedParse accepts. 16-bit / float microscopy is per-image
    min-max scaled to uint8 (BiomedParse's fixed pixel norm expects 8-bit-like range);
    grayscale -> 3-channel; RGBA -> RGB."""
    from PIL import Image

    a = np.asarray(img)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = np.zeros_like(a, np.uint8) if hi <= lo else ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return Image.fromarray(a, mode="RGB")


def build_model(config, weights):
    """Load the 2D BiomedParse model once (K-independent). Imports are lazy so the module
    loads for --help without the BiomedParse env. Requires BIOMEDPARSE_SRC on sys.path."""
    src = os.environ.get("BIOMEDPARSE_SRC", "/disk1/prusek/incontext/BiomedParse")
    if src not in sys.path:
        sys.path.insert(0, src)
    import torch
    from modeling import build_model as _bm
    from modeling.BaseModel import BaseModel
    from utilities.arguments import load_opt_from_config_files
    from utilities.constants import BIOMED_CLASSES
    from utilities.distributed import init_distributed

    cfg = config or os.path.join(src, "configs/biomedparse_inference.yaml")
    opt = load_opt_from_config_files([cfg])
    opt = init_distributed(opt)
    model = BaseModel(opt, _bm(opt)).from_pretrained(weights).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True)
    return model


def infer_fg(model, img, prompt):
    """One image + one text prompt -> native-size boolean foreground (prob > 0.5)."""
    from inference_utils.inference import interactive_infer_image

    pred = interactive_infer_image(model, to_rgb_pil(img), [prompt])  # [n_prompts,H,W] probs
    prob = np.asarray(pred)
    if prob.ndim == 3:
        prob = prob[0]
    return prob > 0.5


# ------------------------------------------------------------------------------------------
def eval_dataset(name, model, pool, test, max_test=None):
    """Return (mean, pk, per_image, vocab) for one dataset (single K-independent pass, seed=0)."""
    if name not in PROMPTS:
        raise RuntimeError(f"no prompt mapping for {name}")
    prompt, metric_override, vocab = PROMPTS[name]
    if prompt is None:
        raise RuntimeError(f"N/A — no in-vocabulary BiomedParse target (vocab={vocab})")
    spec = PANEL[name]
    pk = primary_key(metric_override)                     # 'iou'->fg_iou, 'cldice'->cldice
    _support, test_pairs = load_dataset(spec, pool, test, seed=0)   # FIXED test == ours
    if max_test is not None:
        test_pairs = test_pairs[:max_test]
    if not test_pairs:
        raise RuntimeError("no test images")
    per_image = []
    for img, gt in test_pairs:
        fg = infer_fg(model, img, prompt)
        if fg.shape != np.asarray(gt).shape[:2]:                    # guard size mismatch
            fg = fg[: gt.shape[0], : gt.shape[1]]
        per_image.append(float(score_prediction(metric_override, fg, gt)[pk]))
        sys.stderr.write("."); sys.stderr.flush()
    sys.stderr.write(f" [{name} {pk}={np.mean(per_image):.3f} prompt='{prompt}' vocab={vocab}]\n")
    return float(np.mean(per_image)), pk, per_image, vocab


def resolve_datasets(arg):
    if arg in ("all", ""):
        return [d for d in PANEL_ORDER if d in PANEL]
    names = [n.strip() for n in arg.split(",") if n.strip()]
    bad = [n for n in names if n not in PROMPTS]
    if bad:
        raise SystemExit(f"unknown/unmapped dataset(s): {bad}; valid = {list(PROMPTS)}")
    return names


def _score_path(score_dir, dataset):
    return os.path.join(score_dir, f"biomedparse__{dataset}.json")


# ------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="BiomedParse text-prompted baseline on the AutoSeg panel")
    ap.add_argument("--datasets", default="all", help="comma keys or 'all'")
    ap.add_argument("--pool", type=int, default=20, help="support-pool size (ignored by BiomedParse; "
                    "only fixes the test slice identically to ours)")
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--config", default=None, help="override configs/biomedparse_inference.yaml")
    ap.add_argument("--weights", default="hf_hub:microsoft/BiomedParse",
                    help="from_pretrained target (gated HF repo, or a local biomedparse_v1.pt path)")
    ap.add_argument("--score_dir", default="results/scores_basekscale/biomedparse")
    ap.add_argument("--smoke", action="store_true", help="score only the first test image per dataset")
    args = ap.parse_args()

    print(f"# BiomedParse baseline | weights={args.weights} | TEXT-PROMPTED foundation model "
          f"(K-INDEPENDENT, support IGNORED, seeds=[0])", flush=True)
    os.makedirs(args.score_dir, exist_ok=True)
    t0 = time.time()
    model = build_model(args.config, args.weights)
    print(f"# loaded BiomedParse in {time.time()-t0:.1f}s", flush=True)

    names = resolve_datasets(args.datasets)
    rows, skipped = [], {}
    for name in names:
        try:
            mean, pk, per_image, vocab = eval_dataset(
                name, model, args.pool, args.test, max_test=1 if args.smoke else None)
            out = dict(method="biomedparse", dataset=name, metric=pk, test_per_seed=len(per_image),
                       seeds=[0], per_image=per_image,
                       prompt=PROMPTS[name][0], vocab=vocab,
                       note="text-prompted foundation model; K-independent; support ignored")
            with open(_score_path(args.score_dir, name), "w") as f:
                json.dump(out, f)
            rows.append((name, pk, mean, vocab, PROMPTS[name][0]))
            print(f"  {name:12s} {pk:8s}={mean:.4f}  vocab={vocab:11s} prompt='{PROMPTS[name][0]}'"
                  f"  -> {_score_path(args.score_dir, name)}", flush=True)
        except Exception as e:  # noqa: BLE001 — never crash the whole run; SKIP with a reason
            skipped[name] = f"{type(e).__name__}: {e}"
            print(f"  SKIP {name}: {skipped[name]}", flush=True)
            traceback.print_exc()

    print("\n===== BiomedParse PANEL SUMMARY (text-prompted foundation model; single K-indep pass) =====",
          flush=True)
    for name, pk, mean, vocab, prompt in rows:
        print(f"  {name:12s} {pk:8s}={mean:.4f}  [{vocab}]  '{prompt}'", flush=True)
    for name, why in skipped.items():
        print(f"  {name:12s} SKIP: {why}", flush=True)
    print("BIOMEDPARSE_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
