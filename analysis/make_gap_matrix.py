r"""Build the query-visibility gap matrix.

Rows are policies, columns are (model, length) cells, each entry is the
macro-accuracy change in points when the question moves inside the compressed
region, at the 20 per cent budget that every cell of the matrix shares. A
dagger marks a contrast whose 95 per cent interval excludes zero.

Reads the reports written by regen_reports.py, so the table cannot drift from
the bootstrap that produced them.

    python make_gap_matrix.py > gap_matrix.tex
"""
import re, io, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(os.path.dirname(HERE), "reports")

# (report stem, column header, group)
COLS = [
    ("4k",        "4K",  "Llama-3.1-8B"),
    ("8k",        "8K",  "Llama-3.1-8B"),
    ("16k",       "16K", "Llama-3.1-8B"),
    ("32k",       "32K", "Llama-3.1-8B"),
    ("qwen4k",    "4K",  "Qwen2.5-3B"),
    ("qwen8k",    "8K",  "Qwen2.5-3B"),
    ("qwen16k",   "16K", "Qwen2.5-3B"),
    ("llama3b4k", "4K",  "Llama-3.2-3B"),
    ("loogle16k", "16K", "LooGLE"),
]

ROWS = ["SnapKV", "PyramidKV", "TOVA", "StreamingLLM", "BUZZ",
        "KeyDiff", "H2O", "Value-aware", "KIVI (quantisation)",
        "Full cache (control)"]

LINE = re.compile(
    r"^\s{2}(?P<name>.+?)\s+@\s+(?P<budget>\d+)%\s+"
    r"(?P<pt>[-+][\d.]+)\s*(?:\[\s*(?P<lo>[-+][\d.]+),\s*(?P<hi>[-+][\d.]+)\])?")


def read(stem):
    """gap at the 20% budget, per policy, from one report."""
    path = os.path.join(REPORTS, "ci_report_%s.txt" % stem)
    if not os.path.exists(path):
        return None
    txt = io.open(path, encoding="utf-8", errors="replace").read()
    # only the query-visibility block
    i = txt.find("query visible minus query hidden")
    if i < 0:
        return None
    block = txt[i:]
    nxt = block.find("\n###", 1)
    if nxt > 0:
        block = block[:nxt]
    out = {}
    for line in block.split("\n"):
        m = LINE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        if name.startswith("Full cache (control)"):
            out["Full cache (control)"] = (float(m.group("pt")), None)
            continue
        if m.group("budget") != "20":
            continue
        lo = m.group("lo")
        hi = m.group("hi")
        sep = lo is not None and (float(lo) > 0 or float(hi) < 0)
        out[name] = (float(m.group("pt")), sep)
    return out


data = {stem: read(stem) for stem, _, _ in COLS}
present = [c for c in COLS if data.get(c[0])]
missing = [c[0] for c in COLS if not data.get(c[0])]
if missing:
    print("%% missing reports, columns omitted: %s" % ", ".join(missing),
          file=sys.stderr)

groups = []
for _, _, g in present:
    if not groups or groups[-1][0] != g:
        groups.append([g, 1])
    else:
        groups[-1][1] += 1

o = []
o.append(r"\begin{table}[tp]")
o.append(r"\centering")
o.append(r"\caption{The query-visibility gap across the whole design matrix: "
         r"change in macro accuracy, in points, when the question is moved "
         r"inside the compressed region, at the 20 per cent budget every cell "
         r"shares. Positive means the policy does better when it can see the "
         r"question. A dagger marks a paired contrast whose 95 per cent "
         r"bootstrap interval excludes zero. The last row is the uncompressed "
         r"model under the same prompt change and is the control: it has no "
         r"policy to exploit the question, so anything it moves by is the "
         r"cost of reordering the prompt and not of query visibility.}"
         r"\label{tab:gapmatrix}")
o.append(r"\footnotesize")
o.append(r"\begin{tabular}{@{}l%s@{}}" % ("r" * len(present)))
o.append(r"\toprule")
hdr = [""]
for g, n in groups:
    hdr.append(r"\multicolumn{%d}{c}{%s}" % (n, g))
o.append(" & ".join(hdr) + r" \\")
# cmidrules
pos = 2
rules = []
for g, n in groups:
    rules.append(r"\cmidrule(lr){%d-%d}" % (pos, pos + n - 1))
    pos += n
o.append("".join(rules))
o.append("Policy & " + " & ".join(h for _, h, _ in present) + r" \\")
o.append(r"\midrule")
for r in ROWS:
    cells = []
    for stem, _, _ in present:
        v = data[stem].get(r)
        if v is None:
            cells.append("--")
            continue
        pt, sep = v
        cells.append("$%+.1f%s$" % (pt, r"^{\dagger}" if sep else ""))
    label = r"\midrule" if r == "Full cache (control)" else None
    if label:
        o.append(label)
    o.append("%s & %s \\\\" % (r, " & ".join(cells)))
o.append(r"\botrule")
o.append(r"\end{tabular}")
o.append(r"\end{table}")
print("\n".join(o))
