#!/usr/bin/env python
"""FINE-TUNED specialist baselines: Cellpose / StarDist / micro-SAM trained on the SAME K support masks.

Why this exists: the natural competitor to a biologist holding K annotated masks is not a specialist
trained on thousands of objects with no support (that is the off-the-shelf reference in
``cellpose_stardist_bench.py`` and ``microsam_bench.py``) -- it is that specialist **fine-tuned on those
very K masks**, which is Cellpose's own documented human-in-the-loop workflow. Without this column the
claim "the only workable option on morphologies no specialist covers" is attackable.

FAIRNESS (baseline-fairness protocol). Each of these was a way to accidentally understate a competitor:
  * The K support shots are drawn with the SAME rng as our method
    (``np.random.default_rng(seed).choice(len(pool), K, replace=False)`` in ``sota_final.py``), from the
    same fixed pool, and scored on the same fixed test split -> the comparison is paired per seed, and
    the emitted ``split_fp`` lets ``stats`` prove the two sides scored identical pictures.
  * Fine-tuning starts from the authors' PRETRAINED generalist checkpoint (what a biologist would do),
    from the SAME checkpoint the off-the-shelf column uses, so the only difference between the two
    columns is the fine-tuning itself and not a model swap.
  * The pretrained variant is chosen PER MODALITY, not fixed: H&E data gets the H&E model. Feeding
    an H&E slide to a fluorescence model is a self-inflicted loss for the baseline, not a finding.
  * A FRESH model is built per (dataset, K, seed) -- no leakage of one seed's fine-tuning into the next.
  * Scores are written in ``sota_final.py``'s json schema so the paper's tables, figures and paired
    statistics read them unchanged.

Run inside the per-library env (cellpose: /disk2/prusek/cellpose4_env, stardist: /disk2/prusek/stardist_env,
micro-SAM: /disk2/prusek/microsam_env).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/disk1/prusek/active-segmenter")
sys.path.insert(0, "/disk1/prusek/active-segmenter/scripts")

# Datasets whose stain is haematoxylin and eosin. Both StarDist and micro-SAM ship a dedicated H&E /
# histopathology model, and the fairness protocol requires the per-modality best model: scoring H&E
# with a fluorescence model would understate the baseline and the resulting margin would be ours by
# construction rather than by merit.
HE_DATASETS = {"monuseg"}

# THE FINE-TUNING BUDGET, and why it is expressed this way.
#
# "Use each library's documented default" sounds like the fair rule and is not one. None of these
# libraries documents a recipe for eight images; their defaults target hundreds to thousands, and at
# K=8 they differ by four orders of magnitude in work done: Cellpose's 100 epochs is 800 image
# passes, StarDist's 800 x 400 x batch 8 is 2.56M patch draws, micro-SAM's 100 epochs is ~2.8 h per
# seed. Handing each baseline a wildly different amount of training is not equal treatment, it just
# uses the same word for three different things.
#
# So the budget is one number, EPOCHS OVER THE K SUPPORT IMAGES, and each library's knob is derived
# from it below. At a given K every baseline then sees every annotated mask the same number of times,
# which is the natural unit of "equal opportunity" when the annotation budget is what is scarce.
# Everything the library authors actually tuned -- optimizer, learning rate, augmentation, loss --
# is left at their values, because overriding those substitutes our judgement for theirs.
#
# The number itself is not a guess: `finetune_budget_sweep.py` measures score against budget and we
# take the plateau. That turns "why 100 epochs?" from an argument into a curve.
def library_budget(backend: str, epochs: int, k: int) -> dict:
    """Map a budget in support-epochs onto each library's own training knob."""
    if backend == "cellpose_ft":
        return dict(n_epochs=epochs)                      # iterates the K images per epoch
    if backend == "stardist_ft":
        # epochs x steps_per_epoch x batch patches; steps = K at batch 1 -> epochs passes over K.
        return dict(epochs=epochs, steps_per_epoch=max(1, k), batch_size=1)
    if backend == "microsam_ft":
        return dict(n_iterations=max(1, epochs * k))      # batch 1, so one iteration = one patch
    raise ValueError(f"no budget mapping for {backend}")


def split_fingerprint(pairs) -> str:
    """Content digest of a loaded split. Copied VERBATIM from ``sota_final.py`` -- it must stay
    byte-compatible, since ``stats`` refuses to pair two records whose fingerprints disagree, and a
    digest computed differently here would silently never match ours."""
    h = hashlib.sha256()
    for image, label in pairs:
        for a in (np.ascontiguousarray(image), np.ascontiguousarray(label)):
            h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())
    return h.hexdigest()[:16]


def draw_support(pool, k, seed):
    """EXACTLY the draw sota_final.py uses, so specialist and Exemplar see identical support shots."""
    idx = np.random.default_rng(seed).choice(len(pool), k, replace=False)
    return [pool[i] for i in idx]


def as_instances(label) -> np.ndarray:
    """Return a per-instance label map.

    Vessel and membrane sets ship BINARY foreground, where every object shares the id 1. Training an
    instance model on that teaches it that the entire foreground is one object, which is a training
    bug rather than a property of the baseline, so binary ground truth is connected-component
    labelled first. Already-instanced maps are passed through untouched.
    """
    lb = np.asarray(label)
    if lb.max() <= 1:
        from skimage.measure import label as cc_label
        return cc_label(lb > 0).astype(np.int32)
    return lb.astype(np.int32)


# --------------------------------------------------------------------------------------- Cellpose

def finetune_cellpose(shots, epochs, lr, _dataset):
    """Fine-tune the pretrained Cellpose generalist on the K shots; returns predict(img)->labels.

    ``n_epochs``, ``learning_rate`` and ``batch_size`` are left at Cellpose's own documented defaults
    (100, 1e-5, 1) so this is the library's recommended fine-tune and not a configuration we tuned.

    ``min_train_masks`` is the one deliberate departure. Its default of 5 DROPS any training image
    holding fewer than five objects, which is silent data loss aimed at a many-object training set:
    on a K-shot budget with few objects per image it deleted the entire support and Cellpose then
    died dividing by an empty training set. Keeping the default would have meant either a crash or,
    worse, fine-tuning on a subset of the K masks our own method got in full.
    """
    import torch
    from cellpose import models

    from cellpose_stardist_bench import cpsam_input

    imgs = [cpsam_input(np.asarray(im)) for im, _ in shots]
    lbls = [as_instances(lb) for _, lb in shots]
    model = models.CellposeModel(gpu=True)                      # cellpose-SAM default pretrained generalist
    before = torch.cat([p.detach().flatten().float().cpu() for p in model.net.parameters()]).clone()
    try:
        from cellpose import train as cptrain
    except ImportError as exc:                                  # narrow: only the import
        raise RuntimeError("cellpose>=4 required for train.train_seg; run this backend inside "
                           "/disk2/prusek/cellpose4_env") from exc
    # The except above deliberately does NOT wrap the call below. It used to also catch
    # AttributeError, which swallowed real failures raised INSIDE training and re-reported them as a
    # confusing missing-method error on a 3.x fallback path that does not exist in this env.
    with tempfile.TemporaryDirectory() as td:
        # save_path defaults to the working directory and every fine-tune dumps a full ~1.2 GB cpsam
        # checkpoint. Across datasets x K x seeds that is tens of gigabytes of write-only files, and
        # it fails outright if the working directory is not writable. We read the weights out of the
        # live net, so the on-disk copy has no consumer.
        cptrain.train_seg(model.net, train_data=imgs, train_labels=lbls,
                          learning_rate=lr, batch_size=1, min_train_masks=0, save_path=td,
                          **library_budget("cellpose_ft", epochs, len(imgs)))
    # train_seg returns a PATH to saved weights, so whether the in-memory net it was handed is the
    # one that actually moved is an assumption, not a guarantee. If it silently did not, this column
    # would be the off-the-shelf model wearing a fine-tuned label, and the paper would report that
    # fine-tuning does not help when in fact it never happened. Cheap to check, fatal to miss.
    after = torch.cat([p.detach().flatten().float().cpu() for p in model.net.parameters()])
    if torch.equal(before, after):
        raise RuntimeError("cellpose fine-tuning left every weight unchanged — the model that will "
                           "predict is still the pretrained one; refusing to score it as fine-tuned")

    def predict(img):
        # batch_size is left at the library default so this is literally the off-the-shelf script's
        # call; it affects throughput only, but matching removes one more thing to argue about.
        out = model.eval(cpsam_input(np.asarray(img)), diameter=None)
        return np.asarray(out[0]).astype(np.int32)
    return predict


# --------------------------------------------------------------------------------------- StarDist

def _stardist_augmenter(x, y):
    """StarDist's documented flip/rotation augmenter.

    Its default is no augmentation at all, which with K<=16 images means the network re-sees the
    same handful of patches for the entire schedule. Every StarDist training example ships this.
    """
    k = np.random.randint(4)
    x, y = np.rot90(x, k), np.rot90(y, k)
    if np.random.rand() > 0.5:
        x, y = np.fliplr(x), np.fliplr(y)
    return x, y


def finetune_stardist(shots, epochs, lr, dataset):
    """Fine-tune a pretrained StarDist model on the K shots; returns predict(img)->labels.

    The training budget here is what separates a fine-tuned baseline from a mislabelled off-the-shelf
    one. StarDist's own recipe is 800 epochs x 400 steps x batch 8 at lr 3e-4. Running it at K steps
    per epoch on Cellpose's 1e-5 meant roughly 800 patches at a thirtieth of the learning rate, about
    three thousand times less training than the library asks for. That would have entered the paper
    as "fine-tuning StarDist does not help", when the real finding would have been that we barely
    trained it. Steps are scaled to K so even K=1 takes a meaningful number of gradient steps, and
    the learning rate now defaults per backend instead of being shared with Cellpose.
    """
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    from cellpose_stardist_bench import _n_tiles, to_gray, to_rgb_uint8

    # H&E nuclei are dark on light; fluorescence nuclei are bright on dark. Scoring H&E with the
    # fluorescence model is both out of domain and polarity inverted. The off-the-shelf script
    # already picks per modality, so the fine-tuned column must too, or its handicap is our
    # model-selection error rather than a property of fine-tuning.
    he = dataset in HE_DATASETS
    variant = "2D_versatile_he" if he else "2D_versatile_fluo"

    def prep(im):
        if he:                                   # the H&E model is n_channel_in=3
            return normalize(to_rgb_uint8(im).astype(np.float32), 1, 99.8, axis=(0, 1))
        return normalize(to_gray(np.asarray(im)).astype(np.float32), 1, 99.8)

    X = [prep(im) for im, _ in shots]
    Y = [as_instances(lb) for _, lb in shots]
    model = StarDist2D.from_pretrained(variant)
    budget = library_budget("stardist_ft", epochs, len(X))
    steps, batch = budget["steps_per_epoch"], budget["batch_size"]
    model.config.train_epochs = epochs
    model.config.train_learning_rate = lr
    model.config.train_steps_per_epoch = steps
    model.config.train_batch_size = batch
    before = [w.copy() for w in model.keras_model.get_weights()]
    # StarDist requires validation data; with a handful of shots the support itself is the only data we
    # may look at, so we validate on it. This CANNOT leak test information (test is never touched here).
    model.train(X, Y, validation_data=(X, Y), epochs=epochs, steps_per_epoch=steps,
                augmenter=_stardist_augmenter)
    # Same reasoning as the Cellpose guard: mutating `model.config` after `from_pretrained` may or
    # may not reach an already-compiled Keras model, and a fine-tune that silently did nothing would
    # be reported as a fine-tuned baseline.
    after = model.keras_model.get_weights()
    if all(np.array_equal(a, b) for a, b in zip(before, after)):
        raise RuntimeError("stardist fine-tuning left every weight unchanged — refusing to score "
                           "the pretrained model as fine-tuned")
    # StarDist documents threshold optimisation as the step AFTER training; without it the
    # fine-tuned network predicts through the PRETRAINED model's prob/nms thresholds, which is a
    # documented capability switched off and a self-inflicted loss for the baseline. It is
    # optimised on the support, the only data anyone is allowed to look at here, so this leaks
    # nothing from the test split.
    model.optimize_thresholds(X, Y)

    def predict(img):
        x = prep(img)
        # The off-the-shelf script tiles; matching it keeps the only difference between the two
        # columns the fine-tuning itself, and avoids an out-of-memory divergence above 1024 px.
        tiles = _n_tiles(x.shape[:2]) + ((1,) if he else ())
        lab, _ = model.predict_instances(x, n_tiles=tiles)
        return np.asarray(lab).astype(np.int32)
    return predict


# --------------------------------------------------------------------------------------- micro-SAM

def finetune_microsam(shots, dataset, work_dir, epochs, lr, device="cuda"):
    """Fine-tune micro-SAM on the K shots and return predict(img)->instance labels.

    Four settings here are load-bearing; the defaults are wrong for this benchmark in ways that
    silently understate the baseline rather than failing:

    * ``checkpoint_path=None``. Passing the pretrained file explicitly (which micro-SAM's own napari
      trainer does) loads the encoder but leaves the UNETR instance decoder RANDOMLY INITIALISED,
      because the decoder weights live under a key that path does not read. At K=1 and a short
      schedule that produces near-useless instance masks -- it would look like a damning result for
      fine-tuned micro-SAM and would in fact be our bug. Passing the model_type alone keeps the
      pretrained decoder.
    * ``with_segmentation_decoder=True``. Automatic instance segmentation refuses to load a
      checkpoint without a decoder, so this is required for the inference path to work at all.
    * ``n_iterations`` instead of the default ``n_epochs=100``, which is a ~2.8 hour run per seed.
    * ``early_stopping=None``. With only K shots the validation loader necessarily reuses the
      training images, so a val metric must never be allowed to drive stopping.
    """
    from micro_sam.training import default_sam_loader, train_sam

    from microsam_bench import prep_image

    model_type = "vit_l_histopathology" if dataset in HE_DATASETS else "vit_b_lm"
    raws, labs = [], []
    for im, lb in shots:
        a = prep_image(im)
        # EVERY support image must have the SAME number of axes. torch_em validates ndim against a
        # single with_channels flag for the whole batch, so a support draw mixing greyscale and RGB
        # images -- which several of these pools do -- fails whichever way the flag is set, and it
        # only shows up once K is large enough for the draw to straddle both kinds. Forcing three
        # channels is also what SAM's encoder consumes internally, so nothing is lost.
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        # micro_sam's (H,W,3) numpy path is broken (it probes `raw_paths.shape` on a list), so
        # colour input must be handed over channels-first with the flag set explicitly.
        raws.append(np.transpose(a, (2, 0, 1)))
        labs.append(as_instances(lb))

    # Patches cannot exceed the smallest support image, and these pools are not shape-uniform.
    smallest = min(min(r.shape[1:]) for r in raws)
    side = max(64, min(512, int(smallest) // 16 * 16))
    common = dict(raw_paths=raws, raw_key=None, label_paths=labs, label_key=None,
                  patch_shape=(side, side), with_segmentation_decoder=True,
                  with_channels=True, batch_size=1, shuffle=True, num_workers=2)
    name = "ft"
    train_sam(name=name, model_type=model_type,
              train_loader=default_sam_loader(**common, is_train=True),
              val_loader=default_sam_loader(**common, is_train=False),
              checkpoint_path=None, with_segmentation_decoder=True,
              n_iterations=library_budget("microsam_ft", epochs, len(shots))["n_iterations"],
              n_objects_per_batch=8, n_sub_iteration=4,
              early_stopping=None, verify_n_labels_in_loader=None,
              save_root=work_dir, device=device, lr=lr)
    ckpt = os.path.join(work_dir, "checkpoints", name, "best.pt")
    if not os.path.exists(ckpt):
        raise RuntimeError(f"micro-SAM fine-tuning produced no checkpoint at {ckpt}")

    from micro_sam.automatic_segmentation import get_predictor_and_segmenter

    from microsam_bench import run_ais

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, checkpoint=ckpt, device=device,
        segmentation_mode="ais", is_tiled=False)
    if type(segmenter).__name__ != "InstanceSegmentationWithDecoder":
        # Silently falling back to the prompt-free mask generator would score a DIFFERENT method
        # under this column, so refuse instead.
        raise RuntimeError(f"expected the AIS decoder segmenter, got {type(segmenter).__name__} — "
                           f"the fine-tuned checkpoint has no decoder_state")

    def predict(img):
        return np.asarray(run_ais(predictor, segmenter, img, 0)).astype(np.int32)
    return predict


BACKENDS = {"cellpose_ft": finetune_cellpose, "stardist_ft": finetune_stardist,
            "microsam_ft": finetune_microsam}

# Each library's OWN documented default learning rate, verified against the installed package (and,
# for Cellpose, against its CLI defaults). Sharing one value across backends is how the StarDist
# column ended up trained at a thirtieth of its recommended rate.
LIBRARY_LR = {"cellpose_ft": 1e-5, "stardist_ft": 3e-4, "microsam_ft": 1e-5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--support", type=int, required=True)          # K
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--test", type=int, default=10000)             # loader caps at what exists
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100,
                    help="fine-tuning budget in EPOCHS OVER THE K SUPPORT IMAGES, mapped onto each\n"
                         "library's own knob by library_budget(). One number for every baseline, "
                         "every dataset and every K; calibrated by finetune_budget_sweep.py")
    # NO shared default. Cellpose documents 1e-5, StarDist documents 3e-4; a single shared value was
    # Cellpose's, so StarDist trained at a thirtieth of its own recommended rate and the resulting
    # column understated it. Each backend now gets its library's own default unless overridden.
    ap.add_argument("--lr", type=float, default=None,
                    help="override the per-backend library default (cellpose 1e-5, stardist 3e-4, "
                         "micro-SAM 1e-5)")
    ap.add_argument("--work_dir", default="/disk1/prusek/microsam_ft_work")
    ap.add_argument("--score_dir", required=True)
    ap.add_argument("--fg-scoring", action="store_true")
    args = ap.parse_args()

    from active_segmenter.eval.registry import PANEL, load_dataset
    from active_segmenter.eval.scoring import primary_key, score_prediction

    from cellpose_stardist_bench import label_to_pred

    lr = args.lr if args.lr is not None else LIBRARY_LR[args.backend]
    print(f"  backend={args.backend} lr={lr:g}"
          f"{' (library default)' if args.lr is None else ' (overridden)'}", flush=True)
    names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [n for n in names if n not in PANEL]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; known: {sorted(PANEL)}")

    os.makedirs(args.score_dir, exist_ok=True)
    for name in names:
        spec = PANEL[name]
        eff = spec.metric if spec.metric == "cldice" else ("fg_iou" if args.fg_scoring else spec.metric)
        pk = primary_key(eff)
        # Also record the dataset's OWN metric. Scoring everything as foreground IoU removes
        # exactly the ability these specialists exist for -- separating touching objects -- so
        # the instance-AP comparison has to be recoverable from the same run.
        native_pk = primary_key(spec.metric) if eff != spec.metric else ""
        pool, test = load_dataset(spec, args.pool, args.test, seed=0)
        if len(pool) < args.support:
            print(f"  [{name}] SKIP: pool {len(pool)} < K={args.support}", flush=True)
            continue
        split_fp = split_fingerprint(test)
        per_image, per_image_native, seeds_used = [], [], []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            shots = draw_support(pool, args.support, seed)
            # A FRESH model per seed. Reusing one would let seed 0's fine-tuning leak into every
            # later seed and collapse the across-seed variance the error bars are meant to show.
            if args.backend == "microsam_ft":
                work = os.path.join(args.work_dir, f"{name}_k{args.support}_s{seed}")
                predict = finetune_microsam(shots, name, work, args.epochs, lr)
            else:
                predict = BACKENDS[args.backend](shots, args.epochs, lr, name)
            # per_image is SEED-MAJOR (seed0's N images, then seed1's N, ...). stats() reshapes
            # (-1, test_per_seed) and averages across seeds per image; image-major has the identical
            # length, reshapes without error, and silently produces a wrong paired vector.
            for im, lb in test:
                lab = predict(im)
                # Enumerate instances exactly as the off-the-shelf script does. Walking
                # range(1, max+1) instead injects an empty mask for every skipped label id, and
                # instance_ap counts those as false positives — an asymmetry that can only ever
                # depress the fine-tuned column.
                fg, instances = label_to_pred(lab)
                inst = instances if "instance_ap" in (eff, spec.metric) else None
                per_image.append(float(score_prediction(eff, fg, np.asarray(lb), inst)[pk]))
                if native_pk:
                    per_image_native.append(
                        float(score_prediction(spec.metric, fg, np.asarray(lb), inst)[native_pk]))
            seeds_used.append(seed)
            print(f"    [{name}] K={args.support} seed{seed} "
                  f"{pk}={np.mean(per_image[-len(test):]):.4f}", flush=True)
        if len(per_image) != len(seeds_used) * len(test):
            raise SystemExit(f"[{name}] {len(per_image)} scores != {len(seeds_used)} seeds x "
                             f"{len(test)} images — refusing to write a record stats cannot reshape")
        out = dict(method=args.backend, dataset=name, metric=pk, test_per_seed=len(test),
                   seeds=seeds_used, per_image=per_image, split_fp=split_fp,
                   protocol=dict(pool=args.pool, test=args.test, support=args.support, split_seed=0),
                   finetuned_on_support=True, epochs=args.epochs, lr=lr)
        if native_pk:
            out["metric_native"] = native_pk
            out["per_image_native"] = per_image_native
        fp = os.path.join(args.score_dir, f"{args.backend}__{name}.json")
        with open(fp, "w") as f:
            json.dump(out, f)
        arr = np.array(per_image).reshape(len(seeds_used), len(test))
        print(f"  [{name}] {args.backend} K={args.support}: {pk} "
              f"{arr.mean(1).mean():.4f}±{arr.mean(1).std():.4f} -> {fp}", flush=True)
    print("SPECIALIST_FT_DONE", flush=True)


if __name__ == "__main__":
    main()
