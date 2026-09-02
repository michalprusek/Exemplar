#!/usr/bin/env python3
"""SUPERSEDED — do NOT use for the ISBI campaign. See `make_table_data.py` / `make_final_kscale.py`.

This predates the score-record contract and the single clean campaign tree, and running it against
the current results would produce a wrong table rather than an error:

* `OURS_ALIAS` below is `head_fusion_adaptive`, an OLD method. The current one is
  `head_fusion_best_cgate_film_nobank`, and the campaign writes it under the job label `ours`.
* `DATASET_METRIC` still lists `microtubules`, the dataset WITHDRAWN from the paper. Regenerating a
  table here would reintroduce it.
* It reads `results/{scores_*}` directory layouts from earlier experiments; the campaign writes one
  clean tree, `results/final10/{method}_k{K}/`.
* Its per-baseline regex log-scrapers exist because micro-SAM, PerSAM, Cellpose and StarDist used to
  print only a mean to stdout. All four now emit the score-record contract
  (`active_segmenter/eval/score_record.py`) and are read directly, with a paired significance test
  behind every number instead of a scraped mean.

Kept only for reference until the campaign's tables are final, then delete it. Nothing imports it.

--- original docstring follows ---

Collect the FINAL SOTA table from all result sources (method + external baselines).

Design note (silent-failure-hardened): every parser distinguishes "source absent" (→ "—", a
genuine not-run/not-applicable) from "source PRESENT but produced nothing / an unexpected value"
(→ a loud ``[WARN]`` on stderr). A number in a paper table must never be silently missing, stale,
or mis-sourced, so format drift, a missing steelman variant, or a metric mismatch all warn rather
than quietly degrade. Emits two markdown tables: (A) new-4 external-baseline comparison, (B)
panel-wide Ours-vs-few-shot. Warnings go to stderr; the tables to stdout."""
import json, math, os, re, glob, sys
import numpy as np

R = os.environ.get("ASG_RESULTS", "/disk1/prusek/active-segmenter/results")

# Our final method alias. Native-classical was DROPPED, so it is the base `head_fusion_adaptive`
# (NOT the `_thin` native-gated variant); superres/refine are set at RUN time (encoded in the
# score DIR, not the filename), which is why method() also warns on stale cross-dir duplicates.
OURS_ALIAS = "head_fusion_adaptive"

# Intrinsic per-dataset metric (raw primary_key as written by sota_final.py) — used to catch a
# score file landing in a slot whose column claims a different metric.
DATASET_METRIC = {
    "spheroid": "fg_iou", "spheroidj": "fg_iou", "rozpad": "fg_iou", "kvasir": "fg_iou",
    "hrf": "cldice", "microtubules": "cldice", "drive": "cldice", "isbi2012em": "cldice",
    "dsb2018": "ap", "monuseg": "ap", "ctc_u373": "ap",
}

def _warn(msg):
    sys.stderr.write(f"[WARN] {msg}\n")


def jmean(path, ds=None):
    """mean of ``per_image`` from a sota_final score json, or None if the file is absent.
    Warns if the file EXISTS but yields no values or carries the wrong metric for ``ds``."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    pi = d.get("per_image", [])
    if not pi:
        _warn(f"{path}: present but 'per_image' is empty/missing → dropped to '—'")
        return None
    if ds and d.get("metric") and d["metric"] != DATASET_METRIC.get(ds):
        _warn(f"{path}: metric={d['metric']!r} but {ds} expects {DATASET_METRIC.get(ds)!r}")
    v = float(np.mean(pi))
    if not math.isfinite(v):
        _warn(f"{path}: non-finite mean ({v}) → dropped to '—'")
        return None
    return v

# ---------- METHOD (authoritative) ----------
_METHOD_DIRS = ["scores_final4", "scores_refine", "scores_ctc1024", "scores"]

def method(ds):
    """Our number + provenance dir. Config (superres/refine) is encoded in the DIR, so a stale
    duplicate across dirs is a silent-wrong risk → warn and use the first (highest-priority) dir."""
    hits = [(jmean(f"{R}/{d}/{OURS_ALIAS}__{ds}.json", ds), d) for d in _METHOD_DIRS]
    hits = [(v, d) for v, d in hits if v is not None]
    if len(hits) > 1:
        _warn(f"Ours {ds}: {len(hits)} score files across dirs {[d for _, d in hits]}; "
              f"using {hits[0][1]} (verify it is the intended config)")
    return hits[0] if hits else (None, None)

# ---------- INSID3 / UniverSeg ----------
def base_method(name, ds):
    if name == "insid3":
        return insid3_best(ds)[0]
    return jmean(f"{R}/scores_base/{name}__{ds}.json", ds)

def insid3_best(ds):
    """FAIR INSID3 = best of its refinement variants per dataset (dense-CRF vs guided-filter):
    dense CRF sharpens blob boundaries but COLLAPSES thin structures, so we give INSID3 the better
    of the two per dataset (transparent steelman). Returns (value, which). WARNS if only ONE
    variant is on disk — the reported number is then NOT a genuine max() and can be a collapsed
    (e.g. drive dense=0.007) value that silently under-sells INSID3."""
    dense = jmean(f"{R}/scores_base/insid3__{ds}.json", ds)             # ASG_CRF=dense (default)
    guided = jmean(f"{R}/scores_insid3_guided/insid3__{ds}.json", ds)   # ASG_CRF=guided
    cands = [(v, k) for v, k in [(dense, "crf"), (guided, "guided")] if v is not None]
    if not cands:
        return (None, None)
    if dense is None or guided is None:
        _warn(f"insid3 steelman {ds}: only one variant present (dense={dense}, guided={guided}) "
              f"— reported value is NOT a max() of both")
    return max(cands, key=lambda t: t[0])

# ---------- micro-SAM (parse PANEL SUMMARY; prefer per-modality BEST model logs) ----------
def _microsam_log(path):
    """Parse a micro-SAM PANEL SUMMARY log → {ds: ap}. Warns if the file exists but no row parsed
    (e.g. a `SKIP <model>: ...` failure or a format change) — else a crashed fair-model run would
    silently revert to the weaker base model."""
    if not os.path.exists(path):
        return None
    out = {}
    for m in re.finditer(r"^\s*(\w+)\s+ap\s+([0-9.]+)±([0-9.]+)\s*$", open(path).read(), re.M):
        out[m.group(1)] = float(m.group(2))
    if not out:
        _warn(f"{path}: present but no 'ap' summary row parsed (SKIP/format drift?)")
    return out

def microsam():
    # FAIR per-modality: monuseg = vit_l_histopathology (H&E specialist), ctc = vit_l_lm (best LM).
    # The vit_b_lm panel log is only a fallback; if a fair-model OVERRIDE log is present it MUST
    # yield its key, else we warn (silently keeping the weak base number would break the fairness
    # claim, in the direction that flatters Ours).
    base = _microsam_log(f"{R}/base_full/microsam_full.log") or {}      # vit_b_lm (both)
    out = dict(base)
    for key, path in [("monuseg", f"{R}/base_full/microsam_monuseg_histo.log"),
                      ("ctc_u373", f"{R}/base_full/microsam_ctc_vitl.log")]:
        fair = _microsam_log(path)
        if fair is None:
            continue                                                   # override not run → keep base
        if key not in fair:
            _warn(f"micro-SAM override {path} present but no '{key}' row → keeping weak base "
                  f"{base.get(key)} instead of the fair model")
        else:
            out[key] = fair[key]
    return out

# ---------- StarDist / Cellpose (cs_*.json) ----------
def cellpose_stardist():
    out = {}  # backend -> {ds: mean}
    files = glob.glob(f"{R}/base_full/cs_*.json")
    if not files:
        _warn(f"{R}/base_full/cs_*.json: no cellpose/stardist result files found")
    for f in sorted(files):                                            # sorted → deterministic
        d = json.load(open(f))
        be = d["backend"]
        if be in out:
            _warn(f"duplicate backend {be!r} across cs_*.json ({f}) — later file overwrites earlier")
        res = d.get("results")
        if not res:
            _warn(f"{f}: no 'results' block for backend {be!r}")
            res = {}
        out[be] = {ds: r["primary_mean"] for ds, r in res.items()}
    return out

# ---------- PerSAM / PerSAM-F (parse PANEL RESULTS table) ----------
def persam():
    out = {"PerSAM": {}, "PerSAM-F": {}}
    p = f"{R}/base_full/persam_full.log"
    if not os.path.exists(p):
        return out
    txt = open(p).read()
    for m in re.finditer(r"^(\w+)\s+\w+\s+([0-9.]+)±[0-9.]+\s+([0-9.]+)±[0-9.]+", txt, re.M):
        out["PerSAM"][m.group(1)] = float(m.group(2))
        out["PerSAM-F"][m.group(1)] = float(m.group(3))
    for m in re.finditer(r"^(\w+)\s+\w+\s+([0-9.]+)±[0-9.]+\s+-\s*$", txt, re.M):  # PerSAM-F='-'
        out["PerSAM"][m.group(1)] = float(m.group(2))
    if not out["PerSAM"]:
        _warn(f"{p}: present but no PerSAM rows parsed (format drift?)")
    return out

# ---------- SAM3 (FAIR = text-PCS best-concept; NOT the oracle exemplar-gt path) ----------
def sam3():
    out = {}
    files = glob.glob(f"{R}/base_full/sam3_text_*.log")
    for f in files:
        ds = os.path.basename(f)[len("sam3_text_"):-4]
        best, started = None, False
        for line in open(f):
            if "mean #inst" in line:      # header row: 'prompt  <pk>  mean #inst'; col2=score col3=#inst
                started = True
                continue
            if "SAM3_BENCH_DONE" in line:
                break
            if started:
                m = re.match(r"^\s*(.+?)\s+([0-9]+\.[0-9]+)\s+([0-9]+\.[0-9]+)\s*$", line)
                if m:
                    v = float(m.group(2))                              # score column (0..1 metric)
                    if not 0.0 <= v <= 1.0:
                        _warn(f"{f}: concept {m.group(1)!r} score {v} out of [0,1] — skipped")
                        continue
                    best = v if best is None else max(best, v)
        if best is None:
            _warn(f"{f}: present but no valid concept score parsed")
        else:
            out[ds] = best
    return out

def fmt(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return "—"
    if not math.isfinite(v):
        _warn(f"non-finite cell value {v!r} → '—'")
        return "—"
    return f"{v:.3f}"

NEW4 = ["drive", "isbi2012em", "monuseg", "ctc_u373"]
METR = {"drive": "clDice", "isbi2012em": "clDice", "monuseg": "AP", "ctc_u373": "AP"}
ms, cs, ps, s3 = microsam(), cellpose_stardist(), persam(), sam3()

print("## (A) New-4 external-baseline SOTA table\n")
print("| Method | " + " | ".join(f"{d}<br>({METR[d]})" for d in NEW4) + " |")
print("|" + "---|" * (len(NEW4) + 1))
rows = [
    ("INSID3 (few-shot)",      lambda d: base_method("insid3", d)),
    ("UniverSeg (few-shot)",   lambda d: base_method("universeg", d)),
    ("PerSAM (few-shot)",      lambda d: ps["PerSAM"].get(d)),
    ("PerSAM-F (few-shot)",    lambda d: ps["PerSAM-F"].get(d)),
    ("SAM3 text-PCS (few-shot)", lambda d: s3.get(d)),
    ("micro-SAM AIS (foundation)", lambda d: ms.get(d)),
    ("StarDist-HE (specialist)",   lambda d: cs.get("stardist_he", {}).get(d)),
    ("StarDist-fluo (specialist)", lambda d: cs.get("stardist_fluo", {}).get(d)),
    ("Cellpose-cyto3 (specialist)", lambda d: cs.get("cellpose_cyto3", {}).get(d)),
    ("Cellpose-cpsam (specialist)", lambda d: cs.get("cellpose_cpsam", {}).get(d)),
]
for name, fn in rows:
    print(f"| {name} | " + " | ".join(fmt(fn(d)) for d in NEW4) + " |")
print(f"| **Ours** (+refine) | " + " | ".join(fmt(method(d)[0]) for d in NEW4) + " |")

print("\n## (B) Panel-wide: Ours vs few-shot SOTA (11 datasets)\n")
PANEL = ["spheroid", "spheroidj", "dsb2018", "rozpad", "kvasir", "hrf", "microtubules"] + NEW4
PMETR = {"spheroid": "fg-IoU", "spheroidj": "fg-IoU", "dsb2018": "AP", "rozpad": "fg-IoU",
         "kvasir": "fg-IoU", "hrf": "clDice", "microtubules": "clDice",
         "drive": "clDice", "isbi2012em": "clDice", "monuseg": "AP", "ctc_u373": "AP"}
print("| Dataset | Metric | Ours | INSID3 (variant) | UniverSeg |")
print("|---|---|---|---|---|")
for d in PANEL:
    mv, _src = method(d)
    iv, iwhich = insid3_best(d)                 # surface WHICH refine variant won (auditability)
    uv = base_method("universeg", d)
    ilabel = f"{fmt(iv)} ({iwhich})" if iv is not None else "—"
    print(f"| {d} | {PMETR[d]} | {fmt(mv)} | {ilabel} | {fmt(uv)} |")
print("\n(— = source absent: not run / baseline not applicable to that modality. Any data-present"
      " parse failure warns on stderr. INSID3 variant = which refinement won the per-dataset max.)")
