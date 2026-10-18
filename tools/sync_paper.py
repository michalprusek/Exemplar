#!/usr/bin/env python3
"""Maintainer tool: bring results/ and paper/ in line with the paper in the research repository (AutoSeg).

    python tools/sync_paper.py --autoseg /path/to/AutoSeg [--work DIR] [--check | --write-comment-edits]

For the authors: AutoSeg is not public. Without it, tests/test_paper.py checks the shipped generators and records
against paper/expected_outputs.sha256.

What the paper reads is determined by running its generators, not by a list kept by hand:

  1. paper/make_paper.sh runs AutoSeg's own generators (scripts/, scripts/c119/) on AutoSeg's results/ under an audit
     hook (sys.addaudithook) that logs every file opened and every glob; their outputs must equal the files in
     AutoSeg/paper/isbi2027 (the reference regeneration).
  2. The record set = every file under AutoSeg/results/ the generators opened. Every glob they ran must match only
     opened files (a guard that counts files it does not open would otherwise see a different directory).
  3. results/ is made exactly that set (new and changed files copied, files no generator reads removed; MANIFEST.md
     kept), each JSON record without the fields that describe the machine it ran on (RECORD SCRUB below), and the
     generators are copied into paper/ with their default paths adapted (ADAPT below; each replacement must match
     exactly once, so a changed generator is refused rather than half-adapted) and with internal notes rewritten in
     their comments and docstrings (COMMENT SCRUB below). A generator importing a file not in GENERATORS is refused.
  4. paper/make_paper.sh runs the copied generators on the copied records, again under the audit hook: it may read
     nothing outside results/ and paper/ (and AutoSeg's qualitative dumps, which are not shipped), and every output
     must be byte-identical to AutoSeg/paper/isbi2027 (the creation date matplotlib writes into kscale.pdf excepted).
  5. Every directory of results/ must have a line in results/MANIFEST.md (written by hand: what the arm is, which
     table or figure reads it); a missing line fails the run.
  6. paper/expected_outputs.sha256 is written from the paper's files (normalised: the PDF creation date removed, and
     in .tex comments the relative way to results/, which depends on the output directory), so tests/test_paper.py can
     check a regeneration without the research repository.

--check runs steps 1, 2, 4 and 5 against the repository as it is (no copying) and exits 1 on any difference: the test
that the shipped records and generators still reproduce the paper. Step 4 writes paper/out/ (git-ignored); scratch
files go to --work (default <repo>/../tmp/sync_paper, never /tmp).

--write-comment-edits records the comment and docstring rewrites made by hand in paper/ (against AutoSeg's generators
after ADAPT) in tools/paper_comment_edits.json, after checking that nothing but comments and docstrings differs and
that no internal note is left. When a sync refuses because AutoSeg changed a generator, copy the new generator over
the shipped one, rewrite its flagged comments by hand, and run this.

Also the audit wrapper itself: python tools/sync_paper.py --audit-exec <log.jsonl> <script> [args...]
"""
from __future__ import annotations

import argparse
import ast
import difflib
import filecmp
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tokenize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the generator files, relative to AutoSeg/scripts and to paper/
GENERATORS = ["make_semantic_tables.py", "make_stats.py", "make_ablation_table.py", "make_final_kscale.py",
              "make_fig1_thumbs.py", "render_qual_grid.py", "make_timing_numbers.py", "c119/timing_table.py",
              "c119/timing_common.py"]
# (file, pattern, new): the only code edits to the copied generators -- default paths that point into the research
# repository or a cluster (each pattern must match exactly once). Everything make_paper.sh runs is set through the
# environment or the command line anyway.
ADAPT = [
    ("make_final_kscale.py", r'_REPO = os\.environ\.get\("ASG_REPO_ROOT", "[^"]*"\)',      # a path on the cluster
     '_REPO = os.environ.get("ASG_REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))'),
    ("make_final_kscale.py",
     re.escape('out = os.environ.get("ASG_KSCALE_OUT", f"{_REPO}/paper/isbi2027/figures/kscale.pdf")'),
     'out = os.environ.get("ASG_KSCALE_OUT", f"{_REPO}/paper/out/figures/kscale.pdf")'),
    ("make_timing_numbers.py", re.escape('OUT = os.environ.get("ASG_SEM_OUT", "paper/isbi2027")'),
     'OUT = os.environ.get("ASG_SEM_OUT", "paper/out")'),
]
# COMMENT SCRUB. The generators' comments and docstrings carry the research project's internal notes: who decided
# what and when, entry codes of its experiment registry, cluster host names and paths, quotations in Czech. The
# shipped copies get neutral rewrites by hand, recorded in COMMENT_EDITS as {generator: [edit, ...]} against AutoSeg's
# generator after ADAPT: an edit replaces the block of `old_lines` lines whose SHA-256 is `old_sha256` (it must occur
# exactly once) by `new`. The removed text is stored only as its hash, so the notes do not reach the repository this
# way either. Two guards on every copy: apart from comments and docstrings the token stream is unchanged (so no string
# literal and no code line can differ; strings that reach the generated .tex files keep their text), and no comment or
# docstring matches INTERNAL.
COMMENT_EDITS = os.path.join(REPO, "tools", "paper_comment_edits.json")
INTERNAL = re.compile(r"first author|co-author|\(user\b|\buser (?:decision|directive)\b|\bC\d{2,3}\b|(?i:registry)|"
                      r"addend|\badd\. ?\d|\btulen\b|\bkajman\b|\bpanda\b|/disk\d|/data\d|/scratch|/home/|prusek|"
                      r"CLAUDE\.md|\b\d{1,2}:\d{2}\b|\b\d{1,2} Oct\b|[ěščřžýáíéůúťďňĚŠČŘŽÝÁÍÉŮÚŤĎŇ]")
# RECORD SCRUB. Fields of the score records that describe the machine a run happened on rather than the run (host,
# process, environment variables, interpreter and code paths, the GPU's process list with other users' PIDs) are
# dropped, at any depth; in every other string an absolute path becomes <path>/<last component>, or <path> for a path
# of one or two components (a data or home root). No generator reads these fields (timing_table.py reads method,
# dataset, device, seed, smoke, mode, infer_settings, fit_s, fit_kind, infer_s, infer_mean_s, fit_detail's refit time
# and zoom flag), and step 4 compares the outputs byte for byte after the scrub.
ENV_KEYS = {"host", "pid", "cuda_visible_devices", "env", "env_python", "active_segmenter", "run_env"}
ENV_KEY_PREFIXES = ("gpu_state_",)
ABS_PATH = re.compile(r"(?<![\w.~/>-])/(?:[\w.@+-]+/)*[\w.@+-]+")   # not inside a relative path, a URL or <path>/x
# the files make_paper.sh writes that the manuscript inputs, compared with AutoSeg/paper/isbi2027
OUTPUTS = ["tab_fewshot.tex", "tab_ablation.tex", "numbers_stats.tex", "numbers_ablation.tex", "numbers_fitinf.tex",
           "numbers_timing.tex", "fig_kscale.tex", "figures/kscale.pdf", "figures/kscale.png", "figures/fig1_input.png",
           "figures/fig1_mask.png", "figures/qual3/*.png"]


# ---------------------------------------------------------------------------------------------------------------------
# the audit wrapper

def audit_exec(log_path, script, args):
    """Run `script` as __main__ (its directory first on sys.path, as `python script` does) with an audit hook that logs
    every file opened, glob and directory listing as JSON lines."""
    log = open(log_path, "a", buffering=1)

    def real(x):
        try:
            return os.path.realpath(os.fsdecode(x))
        except (TypeError, ValueError):
            return None

    def hook(event, a):
        if event == "open" and a and not isinstance(a[0], int):
            p = real(a[0])
            if p:
                log.write(json.dumps({"e": "open", "p": p, "m": str(a[1]), "f": a[2]}) + "\n")
        elif event in ("glob.glob", "glob.glob/2"):
            log.write(json.dumps({"e": "glob", "p": os.path.abspath(os.fsdecode(a[0]))}) + "\n")
        elif event == "os.listdir" and a and a[0] is not None and not isinstance(a[0], int):
            log.write(json.dumps({"e": "listdir", "p": real(a[0])}) + "\n")
    sys.addaudithook(hook)
    sys.argv = [script] + list(args)
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    import runpy
    runpy.run_path(script, run_name="__main__")


# ---------------------------------------------------------------------------------------------------------------------

def run_make_paper(repo_root, scripts, out, log, qual_dir=None):
    """paper/make_paper.sh with the given records root and generators, every generator under the audit hook; qual_dir:
    the qualitative dumps (QUAL_DIR), when they are not under the records root."""
    if os.path.exists(log):
        os.remove(log)
    env = dict(os.environ, EXEMPLAR_REPO=repo_root, EXEMPLAR_SCRIPTS=scripts, PYTHON=sys.executable,
               RUN=f"{sys.executable} {os.path.abspath(__file__)} --audit-exec {log}")
    for k in list(env):
        if k.startswith("ASG_") or k == "QUAL_DIR":     # nothing of the caller's shell may steer a generator
            del env[k]
    if qual_dir:
        env["QUAL_DIR"] = qual_dir
    r = subprocess.run(["bash", os.path.join(REPO, "paper", "make_paper.sh"), out], env=env, capture_output=True,
                       text=True)
    if r.returncode:
        sys.exit(f"make_paper.sh failed ({scripts}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return [json.loads(x) for x in open(log)]


def is_write(e):
    m = e["m"]
    if m != "None":
        return any(c in m for c in "wax+")
    return bool(e["f"] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))


def strip_pdf_date(b):
    return re.sub(rb"/CreationDate \([^)]*\)", b"", b)


def normalised(path):
    """A generated file's bytes without what depends on the run rather than the records: the creation date of a PDF,
    and in the comment lines of a .tex the way a path reaches results/ (numbers_timing.tex names its sources relative
    to the output directory, ../../results/... from paper/out)."""
    b = open(path, "rb").read()
    if path.endswith(".pdf"):
        return strip_pdf_date(b)
    if path.endswith(".tex"):
        return b"".join(re.sub(rb"[^\s(,]*results/", b"results/", x) if x.startswith(b"%") else x
                        for x in b.splitlines(keepends=True))
    return b


def expected_listing(ref_dir):
    """`sha256  relative path` of every expected output, normalised, from the paper's own files."""
    lines = []
    for pat in OUTPUTS:
        for rf in sorted(glob.glob(os.path.join(ref_dir, pat))):
            lines.append(f"{hashlib.sha256(normalised(rf)).hexdigest()}  {os.path.relpath(rf, ref_dir)}\n")
    return "".join(lines)


def compare_outputs(out, ref_dir):
    """[(file, verdict)] for every expected output: identical, identical except the PDF creation date, DIFFERENT,
    missing."""
    rows = []
    for pat in OUTPUTS:
        ref_files = sorted(glob.glob(os.path.join(ref_dir, pat)))
        if not ref_files:
            rows.append((pat, "no reference file"))
        for rf in ref_files:
            rel = os.path.relpath(rf, ref_dir)
            mine = os.path.join(out, rel)
            if not os.path.exists(mine):
                rows.append((rel, "MISSING"))
            elif filecmp.cmp(mine, rf, shallow=False):
                rows.append((rel, "identical"))
            elif rel.endswith(".pdf") and strip_pdf_date(open(mine, "rb").read()) == strip_pdf_date(open(rf, "rb").read()):
                rows.append((rel, "identical except /CreationDate"))
            else:
                rows.append((rel, "DIFFERENT"))
        for mine in sorted(glob.glob(os.path.join(out, pat))):
            if not os.path.exists(os.path.join(ref_dir, os.path.relpath(mine, out))):
                rows.append((os.path.relpath(mine, out), "EXTRA (not in the paper)"))
    return rows


# Never shipped: these hold test images of the datasets (the qualitative dumps), which the repository does not
# redistribute; make_paper.sh skips the two figure steps that read them unless the user supplies their own dumps.
NOT_SHIPPED = ("c119/qual_v4/",)


def record_set(events, results_root):
    """Files under results_root that were opened for reading (minus NOT_SHIPPED)."""
    root = os.path.realpath(results_root) + os.sep
    reads = {e["p"] for e in events if e["e"] == "open" and not is_write(e) and e["p"].startswith(root)}
    return {p for p in reads if not p[len(root):].startswith(NOT_SHIPPED)}


def glob_problems(events, link_root, results_root, reads, shipped_root=None):
    """Every glob the generators ran under <link_root>/results must match only files they opened (in results_root),
    and, with shipped_root, match the same relative names there."""
    bad = []
    lr = os.path.join(link_root, "results")
    for pat in sorted({e["p"] for e in events if e["e"] == "glob" and e["p"].startswith(lr + os.sep)}):
        rel_pat = os.path.relpath(pat, lr)
        src = {os.path.relpath(os.path.realpath(m), os.path.realpath(results_root))
               for m in glob.glob(os.path.join(results_root, rel_pat))}
        unread = [m for m in src if os.path.join(os.path.realpath(results_root), m) not in reads]
        if unread:
            bad.append(f"glob {rel_pat} matches files no generator opened: {sorted(unread)[:5]}")
        if shipped_root is not None:
            dst = {os.path.relpath(m, shipped_root) for m in glob.glob(os.path.join(shipped_root, rel_pat))}
            if dst != src:
                bad.append(f"glob {rel_pat}: {len(src)} matches in the source, {len(dst)} in the shipped tree")
    for e in events:
        if e["e"] == "listdir" and e["p"] and e["p"].startswith(os.path.realpath(results_root)):
            bad.append(f"a generator lists {e['p']} directly (not covered by the glob check)")
    return bad


def generator_sources(events, scripts):
    """The generator files AutoSeg's run read (a module import may read only its cached .pyc)."""
    sroot = os.path.realpath(scripts) + os.sep
    out = set()
    for e in events:
        if e["e"] != "open" or is_write(e) or not e["p"].startswith(sroot):
            continue
        rel = os.path.relpath(e["p"], sroot)
        m = re.match(r"(.*)__pycache__/(\w+)\.cpython-\d+\.pyc$", rel)
        out.add(f"{m.group(1)}{m.group(2)}.py" if m else rel)
    return out


def autoseg_commit(autoseg, rel):
    try:
        head = subprocess.run(["git", "-C", autoseg, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", autoseg, "status", "--porcelain", "--", rel], capture_output=True,
                               text=True, check=True).stdout.strip()
        return head + (" + uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown commit"


def path_adapted(autoseg, rel):
    """AutoSeg's generator with ADAPT applied."""
    src = open(os.path.join(autoseg, "scripts", rel)).read()
    for f, pattern, new in ADAPT:
        if f != rel:
            continue
        src, n = re.subn(pattern, lambda m: new, src)
        if n != 1:
            sys.exit(f"REFUSING: {rel} matches `{pattern}` {n} times, not once; update ADAPT")
    return src


def adapted_source(autoseg, rel):
    """The shipped copy of a generator: ADAPT, the comment scrub (checked), and a first line naming the source."""
    src = path_adapted(autoseg, rel)
    out = src
    for ed in comment_edits().get(rel, []):
        lines = out.splitlines(keepends=True)
        n = ed["old_lines"]
        at = [i for i in range(len(lines) - n + 1)
              if hashlib.sha256("".join(lines[i:i + n]).encode()).hexdigest() == ed["old_sha256"]]
        if len(at) != 1:
            sys.exit(f"REFUSING: an edit of {os.path.relpath(COMMENT_EDITS, REPO)} for {rel} matches {len(at)} blocks; "
                     f"copy AutoSeg's generator to paper/{rel}, rewrite its comments and run --write-comment-edits")
        out = "".join(lines[:at[0]] + [ed["new"]] + lines[at[0] + n:])
    problems = scrub_problems(src, out)
    if problems:
        sys.exit(f"REFUSING: paper/{rel} after the comment scrub: " + "; ".join(problems)
                 + ". Rewrite these comments in paper/ by hand and run --write-comment-edits")
    note = (f"# Copied from AutoSeg scripts/{rel} ({autoseg_commit(autoseg, 'scripts/' + rel)}) by tools/sync_paper.py; "
            f"default paths adapted, internal notes rewritten in comments and docstrings.\n")
    lines = out.splitlines(keepends=True)
    at = 1 if lines and lines[0].startswith("#!") else 0
    return "".join(lines[:at] + [note] + lines[at:])


# ---------------------------------------------------------------------------------------------------------------------
# the comment scrub

def split_source(src):
    """(code tokens, comments and docstrings) of a Python source. A docstring is the string that is the first statement
    of the module, a class or a function; code tokens are every other token but line breaks inside a statement."""
    tree = ast.parse(src)
    docs = set()
    owners = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for n in [tree] + [n for n in ast.walk(tree) if isinstance(n, owners)]:
        if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            docs.add((n.body[0].value.lineno, n.body[0].value.col_offset))
    code, notes = [], []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT or (t.type == tokenize.STRING and t.start in docs):
            notes.append(t.string)
        elif t.type != tokenize.NL:
            code.append((t.type, t.string))
    return code, notes, len(docs)


def scrub_problems(before, after):
    """What is wrong with `after` as a comment-only rewrite of `before`."""
    cb, _, nb = split_source(before)
    ca, notes, na = split_source(after)
    out = []
    if cb != ca or nb != na:
        out.append("more than comments and docstrings changed")
    for s in notes:
        for m in INTERNAL.finditer(s):
            line = s[s.rfind("\n", 0, m.start()) + 1:].split("\n", 1)[0]
            out.append(f"internal note {m.group(0)!r} in `{line.strip()[:100]}`")
    return out


def comment_edits():
    """{generator: [{"old_lines", "old_sha256", "new"}, ...]} from COMMENT_EDITS (none without the file)."""
    return json.load(open(COMMENT_EDITS)) if os.path.exists(COMMENT_EDITS) else {}


def write_comment_edits(autoseg):
    """COMMENT_EDITS from the generators in paper/ against AutoSeg's after ADAPT, every copy checked: one edit per group
    of changed lines with one line of context, so a pure insertion has an anchor."""
    no_note = lambda t: re.sub(r"^# Copied from AutoSeg .*\n", "", t, count=1, flags=re.M)
    edits, problems = {}, []
    for rel in GENERATORS:
        src = path_adapted(autoseg, rel)
        mine = no_note(open(os.path.join(REPO, "paper", rel)).read())
        problems += [f"paper/{rel}: {p}" for p in scrub_problems(src, mine)]
        a, b = src.splitlines(keepends=True), mine.splitlines(keepends=True)
        for grp in difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(1):
            i1, i2, j1, j2 = grp[0][1], grp[-1][2], grp[0][3], grp[-1][4]
            edits.setdefault(rel, []).append(dict(old_lines=i2 - i1,
                                                  old_sha256=hashlib.sha256("".join(a[i1:i2]).encode()).hexdigest(),
                                                  new="".join(b[j1:j2])))
    if problems:
        sys.exit("REFUSING to write the comment edits:\n  " + "\n  ".join(problems))
    with open(COMMENT_EDITS, "w") as f:
        json.dump(edits, f, indent=1, ensure_ascii=False)
        f.write("\n")
    for rel in GENERATORS:                               # the edits must give back exactly the files in paper/
        if no_note(adapted_source(autoseg, rel)) != no_note(open(os.path.join(REPO, "paper", rel)).read()):
            sys.exit(f"the comment edits do not reproduce paper/{rel}")
    print(f"wrote {os.path.relpath(COMMENT_EDITS, REPO)}: {sum(len(v) for v in edits.values())} edits in "
          f"{len(edits)} generators")


# ---------------------------------------------------------------------------------------------------------------------
# the record scrub

# Internal notes inside a record's free-text fields: references to the research project's experiment registry are cut
# out of the string, and a string that quotes a person or is written in Czech is emptied. Dates and times stay (a
# generator prints a timing file's date). No generator reads these strings, and step 4 checks the outputs byte for byte.
REGISTRY_REF = re.compile(r"\s*\(?(?:registry )?\bC\d{2,3}\b(?:,? ?(?:add(?:endum|\.) ?\d+(?:[-/]\d+)?))?\)?:?")
PERSONAL_NOTE = re.compile(r"first author|co-author|\(user\b|[ěščřžýáíéůúťďňĚŠČŘŽÝÁÍÉŮÚŤĎŇ]")


def scrub_record(obj):
    """A score record without the machine's fields, with shortened absolute paths and without internal notes
    (RECORD SCRUB)."""
    def short(m):
        parts = m.group(0).strip("/").split("/")
        return "<path>" if len(parts) <= 2 else "<path>/" + parts[-1]
    if isinstance(obj, dict):
        return {k: scrub_record(v) for k, v in obj.items()
                if k not in ENV_KEYS and not k.startswith(ENV_KEY_PREFIXES)}
    if isinstance(obj, list):
        return [scrub_record(v) for v in obj]
    if isinstance(obj, str):
        if PERSONAL_NOTE.search(obj):
            return ""
        return REGISTRY_REF.sub("", ABS_PATH.sub(short, obj)).strip()
    return obj


def shipped_record_bytes(src):
    """The bytes results/ ships for the research record `src`: a JSON record scrubbed and written in its own layout
    (indent 1 or compact, as json.dump wrote it), the original bytes when the scrub changes nothing or for any other
    file."""
    raw = open(src, "rb").read()
    if not src.endswith(".json"):
        return raw
    text = raw.decode()
    obj = json.loads(text)
    clean = scrub_record(obj)
    if clean == obj:
        return raw
    body = json.dumps(clean, indent=1) if text.startswith("{\n") else json.dumps(clean)
    if json.dumps(obj, indent=1 if text.startswith("{\n") else None) != text.rstrip("\n"):
        sys.exit(f"REFUSING: {src} is not in json.dump's layout, so the scrubbed copy would change more than the "
                 "scrubbed fields")
    return (body + text[len(text.rstrip("\n")):]).encode()


def manifest_problems(results):
    path = os.path.join(results, "MANIFEST.md")
    text = open(path).read() if os.path.exists(path) else ""
    dirs = sorted({os.path.relpath(os.path.dirname(p), results) for p in
                   glob.glob(os.path.join(results, "**", "*"), recursive=True) if os.path.isfile(p)
                   and os.path.basename(p) != "MANIFEST.md"})
    return [f"results/MANIFEST.md has no line for {d}/" for d in dirs if d != "." and f"`{d}/`" not in text] + \
           [f"results/MANIFEST.md has no line for {os.path.basename(p)}" for p in glob.glob(os.path.join(results, "*"))
            if os.path.isfile(p) and os.path.basename(p) != "MANIFEST.md" and f"`{os.path.basename(p)}`" not in text]


def main():
    if len(sys.argv) > 3 and sys.argv[1] == "--audit-exec":
        audit_exec(sys.argv[2], sys.argv[3], sys.argv[4:])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--autoseg", required=True, help="the research repository (scripts/, results/, paper/isbi2027/)")
    ap.add_argument("--work", default=os.path.join(os.path.dirname(REPO), "tmp", "sync_paper"))
    ap.add_argument("--check", action="store_true", help="verify the repository as it is; copy nothing")
    ap.add_argument("--write-comment-edits", action="store_true",
                    help="record the comment rewrites made by hand in paper/ in tools/paper_comment_edits.json; "
                         "nothing else")
    a = ap.parse_args()
    autoseg = os.path.abspath(a.autoseg)
    if a.write_comment_edits:
        write_comment_edits(autoseg)
        return
    ref_dir = os.path.join(autoseg, "paper", "isbi2027")
    results = os.path.join(REPO, "results")
    work = os.path.abspath(a.work)
    if os.path.realpath(work).startswith(("/tmp", "/private/tmp")):
        sys.exit("--work must not be under /tmp")
    shutil.rmtree(work, ignore_errors=True)
    link = os.path.join(work, "src")                     # <link>/results -> AutoSeg/results, outputs in <link>/paper/out
    os.makedirs(os.path.join(link, "paper"))
    os.symlink(os.path.join(autoseg, "results"), os.path.join(link, "results"))
    problems = []

    # 1. the reference regeneration with AutoSeg's own generators
    ev = run_make_paper(link, os.path.join(autoseg, "scripts"), os.path.join(link, "paper", "out"),
                        os.path.join(work, "audit_autoseg.jsonl"))
    rows = compare_outputs(os.path.join(link, "paper", "out"), ref_dir)
    bad = [r for r in rows if not r[1].startswith("identical")]
    print(f"[1] AutoSeg's generators on AutoSeg's records: {len(rows) - len(bad)} of {len(rows)} outputs as in the paper")
    for r in bad:
        print(f"    {r[0]}: {r[1]}")
    problems += [f"reference regeneration: {r[0]} {r[1]}" for r in bad]
    srcs = generator_sources(ev, os.path.join(autoseg, "scripts"))
    unknown = sorted(srcs - set(GENERATORS))
    if unknown:
        problems.append(f"the generators read files outside GENERATORS: {unknown}")
    writes = sorted({e["p"] for e in ev if e["e"] == "open" and is_write(e) and e["p"].startswith(autoseg + os.sep)})
    if writes:
        problems.append(f"a generator wrote into AutoSeg: {writes}")

    # 2. the record set
    reads = record_set(ev, os.path.join(autoseg, "results"))
    rel_reads = sorted(os.path.relpath(p, os.path.realpath(os.path.join(autoseg, "results"))) for p in reads)
    problems += glob_problems(ev, link, os.path.join(autoseg, "results"), reads)
    size = sum(os.path.getsize(p) for p in reads)
    print(f"[2] record set: {len(rel_reads)} files, {size / 1e6:.1f} MB, in "
          f"{len({os.path.dirname(r) for r in rel_reads})} directories")

    # 3. copy (unless --check)
    shipped = sorted(os.path.relpath(p, results) for p in glob.glob(os.path.join(results, "**", "*"), recursive=True)
                     if os.path.isfile(p) and os.path.basename(p) != "MANIFEST.md")
    if a.check:
        missing = sorted(set(rel_reads) - set(shipped))
        extra = sorted(set(shipped) - set(rel_reads))
        changed = [r for r in sorted(set(rel_reads) & set(shipped)) if shipped_record_bytes(
                   os.path.join(autoseg, "results", r)) != open(os.path.join(results, r), "rb").read()]
        for name, lst in (("missing from results/", missing), ("in results/ but read by no generator", extra),
                          ("differ from AutoSeg's copy (after the record scrub)", changed)):
            if lst:
                problems.append(f"{len(lst)} record(s) {name}: {lst[:8]}")
        no_note = lambda t: re.sub(r"^# Copied from AutoSeg .*\n", "", t, count=1, flags=re.M)
        for rel in GENERATORS:
            mine = os.path.join(REPO, "paper", rel)
            if not os.path.exists(mine) or no_note(open(mine).read()) != no_note(adapted_source(autoseg, rel)):
                problems.append(f"paper/{rel} differs from AutoSeg's (adapted) generator")
    else:
        n_new = n_chg = n_scrubbed = 0
        for r in rel_reads:
            src, dst = os.path.join(autoseg, "results", r), os.path.join(results, r)
            data = shipped_record_bytes(src)
            n_scrubbed += data != open(src, "rb").read()
            if os.path.exists(dst) and open(dst, "rb").read() == data:
                continue
            n_new += not os.path.exists(dst)
            n_chg += os.path.exists(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(data)
        removed = sorted(set(shipped) - set(rel_reads))
        for r in removed:
            os.remove(os.path.join(results, r))
        for d in sorted({os.path.dirname(os.path.join(results, r)) for r in removed}, key=len, reverse=True):
            while d.startswith(results + os.sep) and os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
                d = os.path.dirname(d)
        for rel in GENERATORS:
            dst = os.path.join(REPO, "paper", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "w").write(adapted_source(autoseg, rel))
        print(f"[3] results/: {n_new} new, {n_chg} changed, {len(removed)} removed ({n_scrubbed} records scrubbed); "
              f"paper/: {len(GENERATORS)} generators copied ({len(ADAPT)} path adaptations, "
              f"{sum(len(v) for v in comment_edits().values())} comment edits)")

    # 4. the shipped generators on the shipped records, into paper/out (numbers_timing.tex names its sources relative
    #    to the output directory, ../../results from there, as in the paper)
    out = os.path.join(REPO, "paper", "out")
    qual = os.path.realpath(os.path.join(autoseg, "results", NOT_SHIPPED[0]))    # Fig. 1 and 3 from AutoSeg's dumps
    ev2 = run_make_paper(REPO, os.path.join(REPO, "paper"), out, os.path.join(work, "audit_shipped.jsonl"),
                         qual_dir=qual if os.path.isdir(qual) else None)
    rows = compare_outputs(out, ref_dir)
    bad = [r for r in rows if not r[1].startswith("identical")]
    print(f"[4] shipped generators on shipped records: {len(rows) - len(bad)} of {len(rows)} outputs as in the paper")
    for r in rows:
        print(f"    {r[0]}: {r[1]}")
    problems += [f"shipped regeneration: {r[0]} {r[1]}" for r in bad]
    outside = sorted({e["p"] for e in ev2 if e["e"] == "open" and not is_write(e)
                      and e["p"].startswith(os.path.dirname(REPO) + os.sep)
                      and not e["p"].startswith((os.path.join(REPO, "results") + os.sep,
                                                 os.path.join(REPO, "paper") + os.sep, out + os.sep, qual + os.sep))})
    if outside:
        problems.append(f"the shipped regeneration read files outside results/ and paper/: {outside[:8]}")
    reads2 = record_set(ev2, results)
    problems += glob_problems(ev2, REPO, results, reads2, shipped_root=results)
    if not a.check and reads2 != {os.path.join(os.path.realpath(results), r) for r in rel_reads}:
        problems.append("the shipped regeneration read another record set than AutoSeg's")

    # 5. the manifest; 6. the expected outputs for tests/test_paper.py
    problems += manifest_problems(results)
    exp_path = os.path.join(REPO, "paper", "expected_outputs.sha256")
    listing = expected_listing(ref_dir)
    if a.check:
        if not os.path.exists(exp_path) or open(exp_path).read() != listing:
            problems.append("paper/expected_outputs.sha256 does not list the paper's current files")
    else:
        open(exp_path, "w").write(listing)
    digest = hashlib.sha256("".join(f"{r}\n" for r in rel_reads).encode()).hexdigest()[:12]
    print(f"[5] manifest checked; record list digest {digest}")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
