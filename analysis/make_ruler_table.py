r"""
Read one or more RULER kv_bench_results.jsonl files and emit the LaTeX tables
for the RULER subsection.

    python make_ruler_table.py kaggle_out_ruler/kv_bench_results.jsonl [more ...]

Files are merged and de-duplicated on (model, ruler_config, condition, task,
method, ratio, trial), LAST FILE WINS, so a corrected rerun layers over an
earlier sweep by being passed last. One RULER context length per invocation:
ruler_config is in the key so a mixed call keeps both lengths' rows, and the
script then refuses them, because every aggregate here ignores the length. ruler_load() takes the FIRST RULER_SAMPLES rows of each task
in dataset order, so trial N names the same sample in every run -- that is what
makes the merge meaningful rather than a pile of unrelated rows.

Why this is not make_table.py. The synthetic sweep scores a binary hit and
averages over needle depth. RULER scores a FRACTION (string_match_all gives
partial credit when only some references are found) and has no depth axis at
all; its second dimension is the task. Averaging the two kinds of number
together would be meaningless, so they stay in separate scripts.

Every task mean is computed first, then tasks are averaged with equal weight
(a MACRO average). Micro-averaging would silently weight the tasks whose
references are more numerous, and cwe/fwe/vt have many references per sample
while niah_single_* have one -- so a micro average would mostly report
performance on three of the thirteen tasks.

Produces, in sn-jnl style (\toprule / \midrule / \botrule):
  1. tab:ruler       macro accuracy against retained-cache fraction, per method
  2. tab:rulertask   per-task accuracy at the tight budget -- the disjointness
  3. tab:rulercond   query-visible minus query-hidden, if both conditions exist
  4. tab:rulermodel  per-model comparison, if more than one model is present
plus a plain-text summary for sanity-checking before pasting anything.
"""
import sys, json, re, collections

PATHS = sys.argv[1:] or ["kaggle_out_ruler/kv_bench_results.jsonl"]

NICE = {
    "full":         "Full cache (no compression)",
    "streamingllm": "StreamingLLM",
    "h2o":          "H2O",
    "snapkv":       "SnapKV",
    "pyramidkv":    "PyramidKV",
    "tova":         "TOVA",
    "keydiff":      "KeyDiff",
    "valueaware":   "Value-aware",
    "buzz":         "BUZZ",
    "kivi":         "KIVI (quantisation)",
}
CITE = {
    "full":         "",
    "streamingllm": r"\cite{xiao2024streamingllm}",
    "h2o":          r"\cite{zhang2023h2o}",
    "snapkv":       r"\cite{li2024snapkv}",
    "pyramidkv":    r"\cite{cai2024pyramidkv}",
    "tova":         r"\cite{oren2024tova}",
    "keydiff":      r"\cite{park2025keydiff}",
    "valueaware":   r"\cite{guo2024valuematters}",
    "buzz":         r"\cite{zhao2026buzz}",
    "kivi":         r"\cite{liu2024kivi}",
}
ORDER = list(NICE)

# RULER's own task order: single-needle, multi-key, multi-value/query,
# then the aggregation and QA tasks that are not needle retrieval at all.
TASK_NICE = [
    ("niah_single_1",   "S-NIAH-1"),
    ("niah_single_2",   "S-NIAH-2"),
    ("niah_single_3",   "S-NIAH-3"),
    ("niah_multikey_1", "MK-NIAH-1"),
    ("niah_multikey_2", "MK-NIAH-2"),
    ("niah_multikey_3", "MK-NIAH-3"),
    ("niah_multivalue", "MV-NIAH"),
    ("niah_multiquery", "MQ-NIAH"),
    ("vt",              "VT"),
    ("cwe",             "CWE"),
    ("fwe",             "FWE"),
    ("qa_1",            "QA-1"),
    ("qa_2",            "QA-2"),
]

# load+merge
merged = {}
for path in PATHS:
    n_before = len(merged)
    n_read = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("taskset") != "ruler":
                continue
            n_read += 1
            key = (r.get("model", ""), str(r.get("ruler_config", "")),
                   r["condition"], r["task"],
                   r["method"], r["ratio"], r["trial"])
            merged[key] = r
    print("# %-52s %5d ruler rows, %+5d new"
          % (path, n_read, len(merged) - n_before), file=sys.stderr)

rows = list(merged.values())
if not rows:
    sys.exit("no RULER rows found -- are these synthetic-sweep files?")

MODELS = sorted({r.get("model", "") for r in rows})
CONDS = sorted({r["condition"] for r in rows})
TASKS = [t for t, _ in TASK_NICE if any(r["task"] == t for r in rows)]
RATIOS = sorted({r["ratio"] for r in rows if r["method"] != "full"}, reverse=True)
METHODS = [m for m in ORDER if any(r["method"] == m for r in rows)]
CONFIG = sorted({str(r.get("ruler_config", "")) for r in rows})

# One RULER context length per invocation. Every aggregate below ignores the
# length, so mixing two would average them under a caption naming one.
if len(CONFIG) > 1:
    sys.exit("refusing to mix RULER context lengths in one table: %s\n"
             "Every aggregate here ignores the length and the caption names "
             "only the first, so the output would be a silent average across "
             "lengths. Pass one length at a time." % ", ".join(CONFIG))


def cell(model, cond, method, ratio, task):
    """Mean score for one cell, or None when the cell was never run."""
    v = [r["score"] for r in rows
         if r.get("model", "") == model and r["condition"] == cond
         and r["method"] == method and r["task"] == task
         and (method == "full" or r["ratio"] == ratio)]
    return sum(v) / len(v) if v else None


def macro(model, cond, method, ratio, tasks=None):
    """Macro mean over tasks: per-task mean first, then equal weight each."""
    per = [cell(model, cond, method, ratio, t) for t in (tasks or TASKS)]
    per = [p for p in per if p is not None]
    return 100.0 * sum(per) / len(per) if per else None


def fmt(x, width=5):
    return (" " * width if x is None else "%*.1f" % (width, x))


# Build the primary tables from the most complete model, not whichever sorts
# first. A partial run would otherwise become the headline table with blank
# rows, and a blank cell reads as not applicable rather than not measured.
def _coverage(model):
    n = 0
    for m in METHODS:
        for rt in ([1.0] if m == "full" else RATIOS):
            if any(r.get("model", "") == model and r["method"] == m
                   and (m == "full" or r["ratio"] == rt) for r in rows):
                n += 1
    return n


def _baseline(model):
    """Macro full-cache score, used only to break a coverage tie."""
    per = collections.defaultdict(list)
    for r in rows:
        if r.get("model", "") == model and r["method"] == "full":
            per[r["task"]].append(r["score"])
    return (sum(sum(v) / len(v) for v in per.values()) / len(per)) if per else 0.0


# Tie-break on the full-cache baseline. At equal coverage the headline table
# would otherwise be chosen by dict insertion order. The stronger baseline is
# the defensible pick: a drop under a weak one is confounded with the model
# being unable to do the task at all.
BASE_MODEL = max(MODELS, key=lambda m: (_coverage(m), _baseline(m)))
BASE_COND = "qhidden" if "qhidden" in CONDS else CONDS[0]
_cov = {m.split("/")[-1]: (_coverage(m), round(100 * _baseline(m), 1))
        for m in MODELS}
print("# coverage (cells, baseline%%): %s -> primary table uses %s"
      % (_cov, BASE_MODEL.split("/")[-1]), file=sys.stderr)
out = []

# tab:ruler
short = BASE_MODEL.split("/")[-1]
nsamp = len({r["trial"] for r in rows})
def pretty_len(s):
    """4096 -> 4{,}096 so the caption reads as prose, not as a variable name."""
    try:
        return "{:,}".format(int(s)).replace(",", "{,}")
    except (TypeError, ValueError):
        return s or "4{,}096"


CTX = pretty_len(CONFIG[0] if CONFIG else None)

out.append(r"""\begin{table}[t]
\centering
\caption{Accuracy on RULER \citep{hsieh2024ruler} at a %s-token context,
macro-averaged over the %d tasks, for %s. Each cell is %d samples per task.
The question is held outside the compressed region, so no policy has seen it
when it chooses what to keep. Scoring follows the reference implementation:
partial credit for the fraction of references recovered, except on the QA
tasks, which score a hit if any reference is found.}\label{tab:ruler}
\footnotesize
\begin{tabular}{@{}l%s@{}}
\toprule
Method & %s \\
\midrule""" % (CTX, len(TASKS), short, nsamp,
               "c" * (len(RATIOS) + 1),
               " & ".join(["100\\%"] + ["%g\\%%" % (100 * x) for x in RATIOS])))
for m in METHODS:
    cells = []
    full_v = macro(BASE_MODEL, BASE_COND, m, 1.0) if m == "full" else None
    cells.append(fmt(full_v) if m == "full" else "--")
    for rt in RATIOS:
        cells.append("--" if m == "full" else fmt(macro(BASE_MODEL, BASE_COND, m, rt)))
    out.append("%s%s & %s \\\\" % (NICE[m], CITE[m], " & ".join(c.strip() for c in cells)))
    if m == "full":
        out.append(r"\midrule")
out.append(r"""\botrule
\end{tabular}
\end{table}""")
out.append("")

# tab:rulertask
TIGHT = RATIOS[-2] if len(RATIOS) > 1 else RATIOS[-1]
ev = [m for m in METHODS if m not in ("full", "kivi")]
out.append(r"""\begin{table}[t]
\centering
\caption{Per-task accuracy at a %g\%% cache budget, %s. The uncompressed
baseline is given for reference; a task on which the baseline is itself weak
cannot separate the policies. No policy leads everywhere: the best method
changes with the task, which is the disjointness the aggregate in
Table~\ref{tab:ruler} hides.}\label{tab:rulertask}
\scriptsize
\begin{tabular}{@{}l%s@{}}
\toprule
Task & Full & %s \\
\midrule""" % (100 * TIGHT, short, "c" * (len(ev) + 1),
               " & ".join(NICE[m].split(" (")[0].replace("StreamingLLM", "Str.")
                          .replace("PyramidKV", "Pyr.").replace("Value-aware", "Val.")
                          .replace("KeyDiff", "KeyD.").replace("SnapKV", "Snap.")
                          for m in ev)))
for t in TASKS:
    base = cell(BASE_MODEL, BASE_COND, "full", 1.0, t)
    vals = [cell(BASE_MODEL, BASE_COND, m, TIGHT, t) for m in ev]
    best = max([v for v in vals if v is not None], default=None)
    cs = []
    for v in vals:
        if v is None:
            cs.append("")
        elif best is not None and abs(v - best) < 1e-9:
            cs.append(r"\textbf{%.0f}" % (100 * v))
        else:
            cs.append("%.0f" % (100 * v))
    label = dict(TASK_NICE)[t]
    out.append("%s & %s & %s \\\\"
               % (label, "%.0f" % (100 * base) if base is not None else "", " & ".join(cs)))
out.append(r"""\botrule
\end{tabular}
\end{table}""")
out.append("")

# This table must not follow BASE_MODEL. The visibility sweep ran on one model
# and the headline model is chosen on coverage, so when they differ every cell
# here comes out blank, which compiles cleanly and ships silently.
COND_MODELS = [m for m in MODELS
               if any(r.get("model", "") == m and r["condition"] == "qvisible"
                      for r in rows)
               and any(r.get("model", "") == m and r["condition"] == "qhidden"
                       for r in rows)]
# Primary table's model first, so the left column group is the one the
# surrounding prose discusses; any others follow in sorted order. When only one
# model has both conditions this degrades to the old single-group table.
COND_MODELS.sort(key=lambda m: (m != BASE_MODEL, m))

if not COND_MODELS and "qvisible" in CONDS and "qhidden" in CONDS:
    print("# WARNING: both conditions are present in the corpus but no single "
          "model has both, so tab:rulercond is omitted rather than emitted "
          "blank. Run the qvisible condition on one of: %s"
          % ", ".join(m.split("/")[-1] for m in MODELS), file=sys.stderr)

if COND_MODELS:
    print("# tab:rulercond covers %d model(s) with both conditions: %s"
          % (len(COND_MODELS),
             ", ".join(m.split("/")[-1] for m in COND_MODELS)), file=sys.stderr)
    if len(COND_MODELS) < len(MODELS):
        print("# NOTE: %s lack(s) one condition and is omitted from it"
              % ", ".join(m.split("/")[-1] for m in MODELS
                          if m not in COND_MODELS), file=sys.stderr)
    _span = len(RATIOS)
    _hdr = " & ".join(r"\multicolumn{%d}{c}{%s}" % (_span, m.split("/")[-1])
                      for m in COND_MODELS)
    _rule = " ".join(r"\cmidrule(lr){%d-%d}"
                     % (2 + i * _span, 1 + (i + 1) * _span)
                     for i in range(len(COND_MODELS)))
    out.append(r"""\begin{table}[t]
\centering
\caption{Effect of letting the policy see the question. Each entry is accuracy
with the question inside the compressed region minus accuracy with it outside,
in points, macro-averaged over the %d tasks, for each model family on which
both query-visibility conditions were run. Positive means the policy exploits
the question; a value near zero means its importance signal never consults
it.}\label{tab:rulercond}
\footnotesize
\begin{tabular}{@{}l%s@{}}
\toprule
Method & %s \\
%s
& %s \\
\midrule""" % (len(TASKS), "c" * (_span * len(COND_MODELS)), _hdr, _rule,
               " & ".join("%g\\%%" % (100 * x)
                          for _ in COND_MODELS for x in RATIOS)))
    for m in METHODS:
        if m == "full":
            continue
        cs = []
        for mo in COND_MODELS:
            for rt in RATIOS:
                a = macro(mo, "qvisible", m, rt)
                b = macro(mo, "qhidden", m, rt)
                cs.append("" if a is None or b is None else "%+.1f" % (a - b))
        out.append("%s & %s \\\\" % (NICE[m], " & ".join(cs)))
    out.append(r"""\botrule
\end{tabular}
\end{table}""")
    out.append("")

# tab:rulermodel
if len(MODELS) > 1:
    out.append(r"""\begin{table}[t]
\centering
\caption{The same comparison across model families, macro-averaged over the %d
RULER tasks with the question held outside the compressed region. A finding
that survives a change of architecture and tokeniser, not only a change of
scale, is harder to dismiss as an artefact of one
model.}\label{tab:rulermodel}
\footnotesize
\begin{tabular}{@{}l%s@{}}
\toprule
Method & %s \\
\midrule""" % (len(TASKS), "c" * (len(MODELS) * len(RATIOS)),
               " & ".join("\\multicolumn{%d}{c}{%s}" % (len(RATIOS), m.split("/")[-1])
                          for m in MODELS)))
    out.append("& " + " & ".join("%g\\%%" % (100 * x)
                                 for _ in MODELS for x in RATIOS) + r" \\")
    out.append(r"\midrule")
    for m in METHODS:
        if m == "full":
            continue
        cs = [fmt(macro(mo, BASE_COND, m, rt)) for mo in MODELS for rt in RATIOS]
        out.append("%s & %s \\\\" % (NICE[m], " & ".join(c.strip() for c in cs)))
    out.append(r"""\botrule
\end{tabular}
\end{table}""")
    out.append("")

# Length-stamped labels. Four fixed labels would collide once a second length
# enters the same manuscript, and LaTeX resolves a duplicate to the last one.
LSUF = ("%dk" % (int(CONFIG[0]) // 1024)) if CONFIG and CONFIG[0].isdigit() else ""
text = "\n".join(out)
if LSUF:
    # \g<1>, not \1: "\1" + "8k" reads as group 18 and raises.
    text = re.sub(r"(\\(?:label|ref)\{tab:ruler[a-z]*)\}",
                  r"\g<1>" + LSUF + "}", text)
print(text)

# summary
e = sys.stderr
print("\n" + "=" * 72, file=e)
print("models    : %s" % ", ".join(MODELS), file=e)
print("conditions: %s   config: %s" % (", ".join(CONDS), ", ".join(CONFIG)), file=e)
print("tasks     : %d   methods: %d   ratios: %s   rows: %d"
      % (len(TASKS), len(METHODS), RATIOS, len(rows)), file=e)
empty = [r for r in rows if not str(r.get("output", "")).strip()]
print("empty outputs: %d (%.1f%%)%s"
      % (len(empty), 100.0 * len(empty) / len(rows),
         "" if not empty else "  <- " + ", ".join(
             sorted({x["method"] for x in empty}))), file=e)
print("-" * 72, file=e)
for mo in MODELS:
    for cond in CONDS:
        base = macro(mo, cond, "full", 1.0)
        if base is None:
            continue
        print("%s / %s  full baseline = %.1f%%" % (mo.split("/")[-1], cond, base), file=e)
        weak = [(dict(TASK_NICE)[t], 100 * cell(mo, cond, "full", 1.0, t))
                for t in TASKS
                if (cell(mo, cond, "full", 1.0, t) or 0) < 0.8]
        if weak:
            print("   WEAK BASELINE (<80%%, treat with caution): %s"
                  % ", ".join("%s %.0f%%" % w for w in weak), file=e)
print("-" * 72, file=e)
print("per-task winners (eviction policies only):", file=e)
for rt in RATIOS:
    win = collections.Counter()
    for t in TASKS:
        sc = {m: cell(BASE_MODEL, BASE_COND, m, rt, t) for m in ev}
        sc = {k: v for k, v in sc.items() if v is not None}
        if sc:
            win[max(sc, key=lambda k: sc[k])] += 1
    print("  at %-5g %s" % (rt, dict(win)), file=e)
print("=" * 72, file=e)
