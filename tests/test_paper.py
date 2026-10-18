"""(6) paper/make_paper.sh regenerates the paper's tables, number macros and figures from the shipped records, and
every file equals the paper's (paper/expected_outputs.sha256, written by tools/sync_paper.py from the paper's files;
normalised: the PDF creation date removed, and in .tex comments the relative way to results/, which depends on the
output directory). Needs matplotlib, scipy and Pillow; the byte equality of the PNG and PDF figures also needs the
reference versions (matplotlib 3.11.2, Pillow 12.3), so the figures are compared only with those."""
import hashlib
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")


def _normalised(path):
    b = open(path, "rb").read()
    if path.endswith(".pdf"):
        return re.sub(rb"/CreationDate \([^)]*\)", b"", b)
    if path.endswith(".tex"):
        return b"".join(re.sub(rb"[^\s(,]*results/", b"results/", x) if x.startswith(b"%") else x
                        for x in b.splitlines(keepends=True))
    return b


def test_make_paper_reproduces_the_paper(tmp_path):
    import matplotlib
    import PIL
    out = tmp_path / "out"
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASG_")}
    env["PYTHON"] = sys.executable
    r = subprocess.run(["bash", os.path.join(REPO, "paper", "make_paper.sh"), str(out)], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    figures_comparable = matplotlib.__version__ == "3.11.2" and PIL.__version__.startswith("12.3")
    checked = 0
    for line in open(os.path.join(REPO, "paper", "expected_outputs.sha256")):
        digest, rel = line.split()
        if rel.endswith((".png", ".pdf")) and not figures_comparable:
            continue
        if ("fig1_" in rel or "qual3/" in rel) and not os.path.exists(str(out / rel)):
            continue                                     # drawn from dataset images, which the repository does not ship
        assert hashlib.sha256(_normalised(str(out / rel))).hexdigest() == digest, rel
        checked += 1
    assert checked >= 7                                  # at least every .tex file
