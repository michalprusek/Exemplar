#!/usr/bin/env python
"""Merge per-host campaign score trees into the ONE tree the generators and ``stats`` read.

The campaign runs across two machines with separate filesystems -- tulen writes under /disk1 and
kajman under /scratch -- so the records that must be compared against each other never meet until
something puts them in one place. ``stats`` refuses duplicates by design, so that merge cannot be a
``cp -r``: it has to decide what to do when the same cell exists twice.

Three things this checks, none of which a copy would:

* **Genuine conflicts.** The same (label, K, dataset) present in two trees with DIFFERENT contents
  means the same cell was measured twice and the two runs disagree. Refuse and name both paths; a
  merge that silently keeps one is how a number nobody chose reaches the paper. Byte-identical
  duplicates are fine and are skipped quietly, because re-running a cell after a restart is normal.

* **Split identity ACROSS hosts.** Pairing our method against a baseline is only meaningful if both
  scored the same images. Each record carries ``split_fp``, a content digest of the test split, and
  every record for one dataset must agree on it regardless of which machine produced it. The hosts
  hold separate copies of the data, so this is a real risk, not a formality -- and it is invisible
  afterwards, because a mismatched pair is silently SKIPPED by stats rather than reported.

* **Protocol drift.** Records for one dataset that disagree on pool, test size or seed count were
  produced under different protocols and are not comparable even when the split matches.

Nothing is deleted and no source tree is modified: the destination is written fresh, so a bad merge
is undone by removing the destination.

    python scripts/merge_score_trees.py --out results/final_merged \\
        --tree tulen=/path/from/tulen/final10 --tree kajman=/path/from/kajman/final10
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import sys
from collections import defaultdict


def _records(tree):
    """Yield (relative path, absolute path) for every score record one level down in ``tree``."""
    for sub in sorted(os.listdir(tree)):
        d = os.path.join(tree, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                yield os.path.join(sub, f), os.path.join(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="append", required=True, metavar="NAME=PATH",
                    help="a source tree, labelled so the report can say where each record came from")
    ap.add_argument("--out", required=True, help="destination tree (must not already exist)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the destination; the default refuses so a stale merge cannot be "
                         "silently topped up with newer records and read as one campaign")
    args = ap.parse_args()

    trees = []
    for spec in args.tree:
        if "=" not in spec:
            raise SystemExit(f"--tree needs NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        if not os.path.isdir(path):
            raise SystemExit(f"{name}: {path} is not a directory")
        trees.append((name, path))

    if os.path.exists(args.out):
        if not args.force:
            raise SystemExit(f"{args.out} already exists. Refusing: merging into a populated tree "
                             f"mixes campaigns, and the result reads as one. Remove it or pass --force.")
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    placed = {}                       # rel -> (source name, abs path)
    conflicts, copied = [], 0
    for name, path in trees:
        for rel, src in _records(path):
            if rel in placed:
                prev_name, prev = placed[rel]
                if filecmp.cmp(prev, src, shallow=False):
                    continue          # same cell measured twice, identical result: normal after a restart
                conflicts.append((rel, prev_name, prev, name, src))
                continue
            dst = os.path.join(args.out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            placed[rel] = (name, src)
            copied += 1

    if conflicts:
        print(f"REFUSING to merge: {len(conflicts)} cell(s) measured twice with DIFFERENT results.",
              file=sys.stderr)
        for rel, an, ap_, bn, bp in conflicts:
            print(f"  {rel}\n    {an}: {ap_}\n    {bn}: {bp}", file=sys.stderr)
        print("Decide which run is authoritative and remove the other before merging; keeping one "
              "silently would put a number nobody chose in the paper.", file=sys.stderr)
        shutil.rmtree(args.out)
        return 1

    # Cross-host consistency. Done AFTER copying so the report covers the merged tree exactly.
    by_ds = defaultdict(list)
    for rel, (name, _) in placed.items():
        with open(os.path.join(args.out, rel)) as f:
            r = json.load(f)
        if not isinstance(r, dict) or "dataset" not in r:
            continue
        by_ds[r["dataset"]].append((rel, name, r))

    problems = 0
    for ds, recs in sorted(by_ds.items()):
        fps = defaultdict(list)
        for rel, name, r in recs:
            if r.get("split_fp"):
                fps[r["split_fp"]].append(f"{name}:{rel}")
        if len(fps) > 1:
            problems += 1
            print(f"!! {ds}: records disagree on the TEST SPLIT. Any comparison across these is "
                  f"meaningless and stats() will silently skip it:", file=sys.stderr)
            for fp, who in sorted(fps.items()):
                print(f"     {fp}  <- {', '.join(sorted(who)[:4])}"
                      f"{' ...' if len(who) > 4 else ''}", file=sys.stderr)

        protos = defaultdict(list)
        for rel, name, r in recs:
            p = r.get("protocol") or {}
            key = (p.get("pool"), r.get("test_per_seed"), len(r.get("seeds") or []))
            protos[key].append(f"{name}:{rel}")
        if len(protos) > 1:
            problems += 1
            print(f"!! {ds}: records disagree on the PROTOCOL (pool, test size, seed count); they "
                  f"are not comparable even where the split matches:", file=sys.stderr)
            for key, who in sorted(protos.items(), key=lambda kv: str(kv[0])):
                print(f"     pool={key[0]} test={key[1]} seeds={key[2]}  <- "
                      f"{', '.join(sorted(who)[:4])}{' ...' if len(who) > 4 else ''}", file=sys.stderr)

    per_source = defaultdict(int)
    for name, _ in placed.values():
        per_source[name] += 1
    print(f"merged {copied} record(s) into {args.out}")
    for name, n in sorted(per_source.items()):
        print(f"  {n:>4} from {name}")
    labels = sorted({rel.split(os.sep)[0] for rel in placed})
    print(f"  {len(labels)} cell group(s): {', '.join(labels)}")
    if problems:
        print(f"\n{problems} consistency problem(s) above. The tree was still written -- inspect "
              f"before reporting anything from it.", file=sys.stderr)
        return 2
    print("  split fingerprints and protocols agree within every dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
