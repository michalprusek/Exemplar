"""Per-dataset, per-K mean +/- std (sample, ddof=1) on the Table-1 foreground metric, for our method
and every baseline the campaign measured. Instance datasets use foreground IoU (fair to the semantic-only
baselines); vessels and filaments use centreline Dice.

Reads the ONE clean campaign tree, `results/final10/{method}_k{K}/`, and nothing else. It used to search
a prioritised list of hand-named directories from earlier experiments, which is how the K-scaling figure
came to plot one method at K=8 and a different one at K=1/4/16. A missing directory must be a missing
CELL, never a silent substitution from an older run."""
import glob
import json
import os

import numpy as np

_REPO = os.environ.get("ASG_REPO_ROOT", "/disk1/prusek/active-segmenter")
ROOT = os.environ.get("ASG_RESULTS_ROOT", f"{_REPO}/results")
FINAL = "final10"

# The reported panel. `microtubules` is deliberately ABSENT: that dataset was withdrawn from the paper
# and must not appear in any generated table. It was still listed here, so regenerating the table would
# have reintroduced a withdrawn dataset into the manuscript.
FG = {"spheroidj": "fg_iou", "dsb2018": "fg_iou", "monuseg": "fg_iou", "ctc_u373": "fg_iou",
      "drive": "cldice", "hrf": "cldice"}
DATASETS = list(FG)

# The keys below are run_campaign.py's JOB LABELS, which is what names the directories, NOT the backend
# method names that name the FILES inside them. Our own row is `ours`, not
# `head_fusion_best_cgate_film_nobank`; using the latter looks right and silently finds nothing, since
# `find` globs `*__{dataset}.json` and so only the directory name has to match. A smoke test against a
# synthetic tree caught exactly that, printing "--" for every one of our own cells.
OURS = "ours"

# Every method the campaign measures. This list drifted from the campaign once already (the K-scaling
# figure drew four competitors where nine had been measured), so it is written out ONCE, here, and the
# score directory is derived from the key rather than hand-named per method.
METHODS = [("Ours", OURS), ("SegGPT", "seggpt"), ("UniverSeg", "universeg"), ("Tyche", "tyche"),
           ("INSID3", "insid3"), ("Matcher", "matcher"), ("PerSAM", "persam"),
           ("Cellpose-FT", "cellpose_ft"), ("StarDist-FT", "stardist_ft"),
           ("microSAM-FT", "microsam_ft"),
           # support-blind: one directory, no _k suffix, identical in every K block
           ("Cellpose", "cellpose_sam"), ("StarDist", "stardist"), ("microSAM", "microsam")]
# Methods that ignore the support masks entirely, so they have no K axis and their directory carries
# no _k suffix. Their numbers repeat across K blocks by construction, which the output states.
K_FREE = {"cellpose_sam", "stardist", "microsam"}
# Methods whose published contribution is a fixed, small support size. run_campaign schedules ONLY
# these K for them, so their absence at any other K is by design, not a failure. Without this the
# MISSING line reports 22 phantom gaps on a fully successful campaign, and the file's own comment
# warns that listing expected absences beside real ones trains the reader to ignore the line.
ONESHOT_KS = {"matcher": {1}, "persam": {1, 8}}   # K=4 Matcher dropped (one-shot; cost)
INSTANCE = {"ctc_u373", "dsb2018", "monuseg"}


def stat(path):
    d = json.load(open(path))
    pi = np.asarray(d["per_image"], float)
    ns, t = len(d["seeds"]), d["test_per_seed"]
    if ns * t != len(pi):                            # FAIL-LOUD: malformed/partial file, refuse to guess
        raise ValueError(f"{path}: per_image len {len(pi)} != {ns}*{t}; malformed score file")
    seedmeans = pi.reshape(ns, t).mean(1)
    # n is returned and PRINTED. A run that died after one seed reports +/-0.000, which reads as an
    # exceptionally reproducible ten-seed measurement rather than as a single sample; and nothing
    # otherwise checks that two methods on the same row were measured over the same number of seeds.
    sd = float(seedmeans.std(ddof=1)) if len(seedmeans) > 1 else float("nan")
    return d["metric"], float(seedmeans.mean()), sd, int(len(seedmeans))


def dirs_for(method_key, ds, k):
    """The single campaign directory for (method, K). One entry, not a fallback chain.

    INSID3 runs BOTH crf modes and is reported at the per-dataset better of the two, which is this
    repo's documented steelman: guided collapses on blobs, dense collapses on thin vessels, so a
    single mode understates it on roughly half the panel.
    """
    if method_key in K_FREE:
        return [f"{FINAL}/{method_key}"]
    if method_key == "insid3":
        return [f"{FINAL}/insid3_guided_k{k}", f"{FINAL}/insid3_dense_k{k}"]
    return [f"{FINAL}/{method_key}_k{k}"]


def find(method_key, ds, k):
    """Best metric-matching cell for (method, dataset, K), or None if the campaign has no such cell.

    For INSID3 both crf modes are read and the BETTER is taken, per the steelman above. For every
    other method there is exactly one directory, so `best` simply picks the only candidate.
    """
    best = None
    for d in dirs_for(method_key, ds, k):
        matches = []
        for f in sorted(glob.glob(os.path.join(ROOT, d, f"*__{ds}.json"))):  # deterministic
            m, mean, sd, n = stat(f)
            if m != FG[ds]:
                continue
            # A dir can hold two records for two VARIANTS (PerSAM writes persam__ AND persam_f__).
            # The record filename carries the harness method name, so the base variant's file starts
            # with the exact method_key while the F-variant does not. Keep only the base variant here;
            # PerSAM-F would be requested under its own method_key ("persam_f") if the table wants it.
            base = os.path.basename(f).startswith(f"{method_key}__")
            matches.append((mean, sd, n, base))
        # Disambiguate a multi-variant dir to the base variant; a genuinely ambiguous dir (two files
        # that both look like the base) is still an error, since silently picking one hides a defect.
        if len(matches) > 1:
            based = [t for t in matches if t[3]]
            if len(based) == 1:
                matches = based
            else:
                raise ValueError(f"ambiguous {ds}/{FG[ds]} in {d}: {len(matches)} metric-matching "
                                 f"files, {len(based)} look like the base variant")
        if matches and (best is None or matches[0][0] > best[0]):
            best = matches[0]
    return best


def expected_absent(method_key, k):
    """Is (method, K) a cell the campaign deliberately never schedules?

    Two causes, both by design: a one-shot method runs only at its documented support sizes, and
    ctc_u373's pool of 15 cannot supply K=16. Everything else absent is a real gap.
    """
    if method_key in ONESHOT_KS and k not in ONESHOT_KS[method_key]:
        return True
    return False


for k in (1, 4, 8, 16):
    print(f"\n================= K = {k} =================")
    print(f"{'dataset':13} {'metric':7} " + " ".join(f"{n:16}" for n, _ in METHODS))
    n_seeds_seen = {}
    for ds in DATASETS:
        row = f"{ds:13} {FG[ds]:7} "
        best = None
        vals = {}
        for n, key in METHODS:
            r = find(key, ds, k)
            vals[n] = r
            if r and (best is None or r[0] > best):
                best = r[0]
        for n, _ in METHODS:
            r = vals[n]
            if r:
                star = "*" if abs(r[0] - best) < 1e-9 else " "
                # n<2 has no standard deviation to report; printing 0.000 there claims perfect
                # reproducibility for a single sample. Show the seed count instead.
                sd = f"{r[1]:.3f}" if r[2] > 1 else f"n={r[2]}"
                row += f"{r[0]:.3f}±{sd}{star}".ljust(17)
                n_seeds_seen.setdefault(r[2], set()).add(n)
            else:
                row += "--".ljust(17)
        print(row)

    # A missing cell prints as "--", indistinguishable from "measured and not applicable" once the
    # table is read out of context. Name the REAL gaps only: listing the campaign's by-design
    # absences beside them (22 of them on a fully successful run) trains the reader to ignore the
    # line, which is worse than not printing it.
    real, by_design = [], []
    for ds in DATASETS:
        for n, key in METHODS:
            if find(key, ds, k) is not None:
                continue
            # ctc_u373's pool of 15 cannot supply K=16, but the support-blind specialists still
            # have a number there, so the excuse applies only to the K-dependent methods.
            pool_capped = (ds, k) == ("ctc_u373", 16) and key not in K_FREE
            (by_design if (expected_absent(key, k) or pool_capped) else real).append(f"{n}/{ds}")
    if real:
        print(f"  MISSING {len(real)} cell(s) at K={k}: {', '.join(real)}")
    if by_design:
        print(f"  ({len(by_design)} absent BY DESIGN at K={k}: one-shot methods outside their "
              f"documented support sizes, and ctc_u373 whose pool cannot supply K=16)")
    if len(n_seeds_seen) > 1:
        # Two methods on one row measured over different seed counts are not comparable error bars.
        print(f"  !! MIXED SEED COUNTS at K={k}: "
              + "; ".join(f"n={n}: {', '.join(sorted(ms))}" for n, ms in sorted(n_seeds_seen.items())))

print(f"\nK-independent columns (support ignored, identical in every K block): "
      f"{', '.join(n for n, key in METHODS if key in K_FREE)}")
print("INSID3 is reported at the per-dataset better of its two CRF modes (documented steelman);\n"
      "that is an upper bound selected on test and the paper must say so.")
# Measured-but-never-tabulated is as much a reporting gap as missing-but-expected, and only the
# second direction had a check. The campaign schedules ten datasets; this table reports six.
_CAMPAIGN_DATASETS = ["spheroidj", "dsb2018", "monuseg", "ctc_u373", "drive", "hrf",
                      "isbi2012em", "fisbe", "bbbc010", "bacteria"]
_untabulated = [d for d in _CAMPAIGN_DATASETS if d not in DATASETS]
if _untabulated:
    print(f"NOT TABULATED (measured by the campaign, absent from this table): "
          f"{', '.join(_untabulated)}")
