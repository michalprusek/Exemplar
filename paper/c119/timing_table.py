# Copied from AutoSeg scripts/c119/timing_table.py (cb71f85) by tools/sync_paper.py; default paths adapted, internal notes rewritten in comments and docstrings.
"""The timing campaign's records -> fit and inference time per method over the eleven datasets.

Writes <json> (the summary, per method and per dataset) and <tex> (macros \\tfit<Key>, \\tinf<Key> for Table 1's last row,
"fit / inference"). Refuses a method with a missing dataset, a smoke record, a record not on the RTX A5000, or a record
whose split draw differs from seed 0. Only the Table 1 records are read: `<Method>__<dataset>.json` with the method
names below (the nnU-Net check record `check-nnU-Net-full-run__*` is reported beside, not averaged).
Usage: python timing_table.py <records dir> <out.json> <out.tex> [--partial]
--partial: a method missing a dataset gets the
cell "\\dots" (no mean over an incomplete set) and the prose ranges are written only when complete; every other refusal
stays. The file then says PARTIAL in its first line.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timing_common as tc  # noqa: E402

# ASG_DROP_DS=ctc_u373 (the paper's setting: CTC-U373 is not reported, because its conditions need the CTC
# organisers' permission for public use outside the challenge): the datasets named here are left out of every median.
DROP = {x for x in os.environ.get("ASG_DROP_DS", "").split(",") if x}
DATASETS = [ds for ds in tc.DATASETS if ds not in DROP]
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"]

METHODS = {   # Table 1 column -> macro key (letters only)
    "SegGPT": "SegGPT", "UniverSeg": "UniverSeg", "INSID3": "INSID", "Tyche": "Tyche", "Exemplar": "Exemplar",
    "Prior-bank RF": "PriorRF", "Convpaint": "Convpaint", "nnU-Net": "nnUNet", "Cellpose-SAM": "CellposeSAM",
    "StarDist": "StarDist", "micro-SAM": "microSAM",
}


def fmt_time(s):
    """Seconds -> a short table string: two significant digits under a minute (milliseconds under a second), whole
    minutes from one to 99, hours with one decimal above."""
    if s is None:
        return "--"
    if s < 0.995:                                    # 0.995 s and above round to "1 s", not "1000 ms"
        return f"{float(f'{s * 1000:.2g}'):g}" + r"\,ms"
    if s < 60:
        x = float(f"{s:.2g}")
        return (f"{x:g}" if x < 10 else f"{x:.0f}") + r"\,s"
    if s < 99.5 * 60:
        return f"{s / 60:.0f}" + r"\,min"
    return f"{s / 3600:.1f}" + r"\,h"


def sig2(s):
    """Seconds -> two significant digits, no unit ('0.94', '6.1', '18', '100')."""
    x = float(f"{s:.2g}")
    return f"{x:g}" if x < 10 else f"{x:.0f}"


def main():
    d, out_json, out_tex = sys.argv[1:4]
    partial = "--partial" in sys.argv[4:]
    # --nnunet_full: nnU-Net's
    # column from the MEASURED full 100-epoch runs (time_nnunet.py --validate_full, records check-nnU-Net-full-run__*)
    # instead of the 5-epoch extrapolation; inference then comes from the fully trained model.
    nn_full = "--nnunet_full" in sys.argv[4:]
    # --ilastik: the real ilastik 1.4.2 (time_ilastik.py, CPU of the A5000 host, card idle) as a column beside
    # Prior-bank RF, for ASG_SEM_ILASTIK=1. --budgets: the shorter Cellpose-SAM fine-tunes (4 to 200 epochs)
    # ("Cellpose-SAM e<N>"), summarised in the json only (no macro: never a Table 1 cell).
    methods = dict(METHODS)
    if "--ilastik" in sys.argv[4:]:
        methods["ilastik"] = "ilastik"
    json_only = set()
    if "--budgets" in sys.argv[4:]:
        for n in (4, 16, 64, 200):
            methods[f"Cellpose-SAM e{n}"] = None
            json_only.add(f"Cellpose-SAM e{n}")
    pending = []
    summary, lines, bad = {}, [], []
    for m, key in methods.items():
        rows = {}
        src = "check-nnU-Net-full-run" if (nn_full and m == "nnU-Net") else m
        for ds in DATASETS:
            p = tc.record_path(d, src, ds)
            if not os.path.exists(p):
                bad.append(f"{m}: {ds} missing")
                continue
            r = json.load(open(p))
            if r.get("smoke"):
                bad.append(f"{p}: smoke record")
            if "A5000" not in str(r.get("device")):
                bad.append(f"{p}: device {r.get('device')!r}")
            if r.get("seed", 0) != 0:
                bad.append(f"{p}: seed {r.get('seed')}")
            if src != m and (r.get("mode") != "validate_full"
                             or (r.get("infer_settings") or {}).get("model_trained_epochs") != 100):
                bad.append(f"{p}: not a measured 100-epoch run (mode {r.get('mode')!r})")
            rows[ds] = dict(fit_s=r.get("fit_s"), infer_mean_s=r["infer_mean_s"], n_infer=len(r["infer_s"]),
                            fit_kind=r.get("fit_kind"))
            if m == "Exemplar":
                fd = r.get("fit_detail") or {}
                rows[ds].update(refit_with_coarse_s=fd.get("refit_with_coarse_s"), zoom_changed=fd.get("zoom_changed"))
        if len(rows) != len(DATASETS):
            if partial:
                bad[:] = [b for b in bad if not (b.startswith(f"{m}: ") and b.endswith(" missing"))]
                pending.append(f"{m} ({len(DATASETS) - len(rows)} missing)")
                if m not in json_only:
                    lines += [rf"\newcommand{{\tfit{key}}}{{\dots}}", rf"\newcommand{{\tinf{key}}}{{\dots}}"]
                lines.append(f"% {m}: PENDING, {len(rows)} of {len(DATASETS)} datasets")
            continue
        fits = [v["fit_s"] for v in rows.values()]
        fit_mean = None if all(f is None for f in fits) else float(np.mean(fits))
        # The table shows MEDIANS over the datasets: image sizes differ more than a hundredfold,
        # so a mean is set by the two or three largest-image datasets for every method. Same statistic for every column;
        # the prose gives the full range. Means are kept in the json.
        fit_med = None if fit_mean is None else float(np.median(fits))
        if fit_mean is not None and any(f is None for f in fits):
            bad.append(f"{m}: fit time missing on some datasets")
        inf_mean = float(np.mean([v["infer_mean_s"] for v in rows.values()]))
        inf_med = float(np.median([v["infer_mean_s"] for v in rows.values()]))
        summary[m] = dict(fit_mean_s=fit_mean, infer_mean_s=inf_mean, fit_median_s=fit_med, infer_median_s=inf_med, fit_kind=sorted({v["fit_kind"] for v in rows.values()}),
                          per_dataset=rows)
        if m not in json_only:
            lines += [rf"\newcommand{{\tfit{key}}}{{{fmt_time(fit_med)}}}",
                      rf"\newcommand{{\tinf{key}}}{{{fmt_time(inf_med)}}}"]
        lines += [f"% {m}: MEDIAN fit {fit_med if fit_med is None else round(fit_med, 2)} s, inference {inf_med:.4f} s per "
                  f"image over {len(rows)} datasets (means {fit_mean if fit_mean is None else round(fit_mean, 2)} s, "
                  f"{inf_mean:.4f} s)"]
    # Ranges over the eleven datasets for the prose (numbers only, the unit is typed in the sentence). Each method's
    # macros are written once that method is complete.
    words = WORDS
    if "Exemplar" in summary:
        ex = summary["Exemplar"]["per_dataset"]
        refit = [v["refit_with_coarse_s"] for v in ex.values()]
        if any(x is None for x in refit) or any(v["zoom_changed"] is None for v in ex.values()):
            bad.append("Exemplar: refit time or zoom_changed missing (run exemplar_records.py on these records)")
        else:
            fit = [v["fit_s"] for v in ex.values()]
            n_zoom = sum(bool(v["zoom_changed"]) for v in ex.values())
            lines += [rf"\newcommand{{\exFitLo}}{{{sig2(min(fit))}}}", rf"\newcommand{{\exFitHi}}{{{sig2(max(fit))}}}",
                      rf"\newcommand{{\exRefitLo}}{{{sig2(min(refit))}}}", rf"\newcommand{{\exRefitHi}}{{{sig2(max(refit))}}}",
                      rf"\newcommand{{\exRefitZoomWord}}{{{words[n_zoom]}}}",
                      rf"\newcommand{{\exFitMed}}{{{sig2(float(np.median(fit)))}}}",
                      "% Exemplar fit (with the support's coarse views) per dataset: "
                      + ", ".join(f"{ds} {v['fit_s']}" for ds, v in ex.items()),
                      "% Exemplar refit (8th mask added to a fit of 7, plus its coarse view) per dataset: "
                      + ", ".join(f"{ds} {v['refit_with_coarse_s']}{' (zoom changed)' if v['zoom_changed'] else ''}"
                                  for ds, v in ex.items())]
            # the range of Exemplar's time per image over the datasets (the table prints the median)
            inf = [v["infer_mean_s"] for v in ex.values()]
            lines += [rf"\newcommand{{\exInfLo}}{{{sig2(min(inf))}}}", rf"\newcommand{{\exInfHi}}{{{sig2(max(inf))}}}",
                      "% Exemplar time per image per dataset: " + ", ".join(f"{ds} {v['infer_mean_s']:.3f}" for ds, v in ex.items())]
    if "nnU-Net" in summary:
        nn = [v["fit_s"] for v in summary["nnU-Net"]["per_dataset"].values()]
        lines += [rf"\newcommand{{\nnFitLoMin}}{{{min(nn) / 60:.0f}}}", rf"\newcommand{{\nnFitHiMin}}{{{max(nn) / 60:.0f}}}",
                  f"% nnU-Net fit ({'measured full runs' if nn_full else 'extrapolated'}, s) per dataset: "
                  + ", ".join(f"{ds} {v['fit_s']}" for ds, v in summary["nnU-Net"]["per_dataset"].items())]
    if "Cellpose-SAM" in summary:
        cp = [v["fit_s"] / 8 for v in summary["Cellpose-SAM"]["per_dataset"].values()]
        cpf = [v["fit_s"] for v in summary["Cellpose-SAM"]["per_dataset"].values()]
        lines += [rf"\newcommand{{\cpFitLoMin}}{{{min(cpf) / 60:.0f}}}", rf"\newcommand{{\cpFitHiMin}}{{{max(cpf) / 60:.0f}}}"]
        lines += [rf"\newcommand{{\cpFitMedMin}}{{{float(np.median(cpf)) / 60:.0f}}}"]      # abstract
        lines += [rf"\newcommand{{\cpEighthLo}}{{{sig2(min(cp))}}}", rf"\newcommand{{\cpEighthHi}}{{{sig2(max(cp))}}}",
                  "% Cellpose-SAM fine-tune / 8 (s), a lower bound of training it further for an eighth of its length: "
                  + ", ".join(f"{ds} {round(v['fit_s'] / 8, 1)}" for ds, v in summary["Cellpose-SAM"]["per_dataset"].items())]
    chk = os.path.join(d, "check-nnU-Net-full-run__dsb2018.json")
    if os.path.exists(chk) and not nn_full:
        c = json.load(open(chk))
        summary["_nnunet_full_run_check"] = c
        # The check: one full 100-epoch run on DSB2018 against the extrapolation of the same draw. The
        # extrapolation leaves out the pauses between epochs (validation, checkpoint) and its epochs ran under another CPU
        # load, so it UNDERSTATES nnU-Net's time; the prose says so with these two numbers.
        ex = summary.get("nnU-Net", {}).get("per_dataset", {}).get("dsb2018", {}).get("fit_s")
        if ex is None or c.get("dataset") != "dsb2018" or "A5000" not in str(c.get("device")):
            bad.append("nnU-Net check: no extrapolated DSB2018 record to compare with, or the check is not on the A5000")
        else:
            lines += [rf"\newcommand{{\nnCheckFullMin}}{{{c['fit_s'] / 60:.0f}}}",
                      rf"\newcommand{{\nnCheckExtrapMin}}{{{ex / 60:.0f}}}",
                      f"% nnU-Net DSB2018: full 100-epoch run {c['fit_s']:.1f} s against extrapolated {ex:.1f} s"]
    if bad:
        raise SystemExit("REFUSING:\n  " + "\n  ".join(bad))
    json.dump(dict(_what="Fit and inference time per method on one RTX A5000, medians (table) and means "
                         f"over the {WORDS[len(DATASETS)]} datasets (seed-0 draw, K=8, first five test images); nnU-Net's fit "
                         + ("measured in full 100-epoch runs" if nn_full else "extrapolated from five epochs")
                         + ("; ilastik on the CPU of the A5000 host (card idle)" if "ilastik" in methods else ""),
                   methods=summary), open(out_json, "w"), indent=1)
    with open(out_tex, "w") as f:
        f.write("% Generated by scripts/c119/timing_table.py from the timing records. Do not hand-edit.\n")
        if pending:
            f.write("% PARTIAL: pending " + ", ".join(pending) + "\n")
        f.write("\n".join(lines) + "\n")
    for m, v in summary.items():
        if not m.startswith("_"):
            print(f"{m:14s} median fit {fmt_time(v['fit_median_s']):>12s}   inference {fmt_time(v['infer_median_s'])}")


if __name__ == "__main__":
    main()
