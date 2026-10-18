"""(7) What the repository ships carries no internal notes of the research project: the comments and docstrings of the
paper's generators (paper/) and the score records (results/) are checked with the same rules tools/sync_paper.py
applies when it copies them (its COMMENT SCRUB and RECORD SCRUB)."""
import glob
import importlib.util
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("sync_paper", os.path.join(REPO, "tools", "sync_paper.py"))
sync_paper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_paper)


def test_generators_comments_and_docstrings_hold_no_internal_notes():
    files = sorted(glob.glob(os.path.join(REPO, "paper", "**", "*.py"), recursive=True))
    assert len(files) == len(sync_paper.GENERATORS)
    hits = []
    for path in files:
        _, notes, _ = sync_paper.split_source(open(path).read())
        hits += [(os.path.relpath(path, REPO), m.group(0)) for s in notes for m in sync_paper.INTERNAL.finditer(s)]
    assert not hits


def test_records_hold_no_environment_fields_absolute_paths_or_internal_notes():
    def walk(o, where, out):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in sync_paper.ENV_KEYS or k.startswith(sync_paper.ENV_KEY_PREFIXES):
                    out.append(f"{where}: key {k}")
                walk(v, where, out)
        elif isinstance(o, list):
            for v in o:
                walk(v, where, out)
        elif isinstance(o, str):
            for rx, what in ((sync_paper.ABS_PATH, "path"), (sync_paper.REGISTRY_REF, "registry reference"),
                             (sync_paper.PERSONAL_NOTE, "personal note")):
                m = rx.search(o)
                if m and m.group(0).strip():
                    out.append(f"{where}: {what} {m.group(0)!r}")
    files = glob.glob(os.path.join(REPO, "results", "**", "*.json"), recursive=True)
    assert files
    bad = []
    for path in files:
        walk(json.load(open(path)), os.path.relpath(path, REPO), bad)
    assert not bad, bad[:10]


def test_docs_hold_no_cluster_paths():
    bad = []
    for path in glob.glob(os.path.join(REPO, "docs", "**", "*"), recursive=True) + [os.path.join(REPO, "README.md")]:
        if os.path.isfile(path) and not path.endswith((".png", ".pdf")):
            for i, line in enumerate(open(path, errors="replace"), 1):
                if any(x in line for x in ("/disk1/", "/disk2/", "/data2/", "/scratch/", "/home/")):
                    bad.append(f"{os.path.relpath(path, REPO)}:{i}")
    assert not bad, bad[:10]
