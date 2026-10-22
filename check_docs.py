# coding=utf-8
"""
Consistency check of the documentation: balanced code fences, well-formed
tables, no duplicated section numbers, no dead links, no reference to a file
that no longer exists.

Run:  python check_docs.py
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

DOCS = ["README.md", "docs/PAPER.md", "docs/EXPERIMENTS.md", "docs/ARCHITECTURE.md",
        "docs/figures/README.md"]
problems = 0

for path in DOCS:
    if not os.path.exists(path):
        print(f"!! {path}: non esiste")
        problems += 1
        continue
    t = open(path, encoding="utf-8").read()
    lines = t.split("\n")

    fences = [i for i, l in enumerate(lines, 1) if l.strip().startswith("```")]
    if len(fences) % 2:
        print(f"!! {path}: fence dispari (righe {fences})")
        problems += 1

    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            cols = {l.count("|") for l in lines[i:j]}
            if len(cols) > 1:
                print(f"!! {path}: tabella riga {i+1} con colonne disuguali {sorted(cols)}")
                problems += 1
            i = j
        else:
            i += 1

    seen: dict = {}
    for k, l in enumerate(lines, 1):
        m = re.match(r"^(#{2,4}) (\d+(?:\.\d+)*)\.? ", l)
        if m:
            seen.setdefault(m.group(2), []).append(k)
    for num, ks in seen.items():
        if len(ks) > 1:
            print(f"!! {path}: numero di sezione duplicato {num} alle righe {ks}")
            problems += 1

    for idx, ch in enumerate(t):
        if unicodedata.category(ch).startswith("Lo") and ord(ch) > 0x2E80:
            print(f"!! {path}: carattere CJK U+{ord(ch):04X} alla posizione {idx}")
            problems += 1

    for m in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", t):
        url = m.group(1).strip()
        if url.startswith(("http", "#", "mailto")):
            continue
        p = os.path.normpath(os.path.join(os.path.dirname(path), url.split("#")[0]))
        if url.split("#")[0] and not os.path.exists(p):
            print(f"!! {path}: link morto -> {url}")
            problems += 1

    # the files mentioned in the backticks must exist (only the ones that look
    # like a path or a script of this repository).  A reference that the text
    # explicitly declares as removed, or that points outside the repository, is
    # exempt; a bare file name is looked up anywhere in the tree.
    for k, line in enumerate(lines, 1):
        context = " ".join(lines[max(0, k - 4):k]).lower()
        exempt = ("rimos" in context or "~/documents" in context
                  or "repository di riferimento" in context)
        for m in re.finditer(r"`([A-Za-z0-9_./-]+\.(?:py|json|ps1|md|txt|cfg))`", line):
            ref = m.group(1)
            if ref.startswith(("http", "www")):
                continue
            cands = [ref, os.path.join("docs", ref), os.path.join(os.path.dirname(path), ref)]
            if not any(os.path.exists(c) for c in cands) and not exempt:
                found = any(os.path.exists(os.path.join(r, os.path.basename(ref)))
                            for r, _d, _f in os.walk("."))
                if not found:
                    print(f"!! {path}:{k}: file citato ma assente -> {ref}")
                    problems += 1

print(f"PROBLEMI: {problems}")
sys.exit(1 if problems else 0)
