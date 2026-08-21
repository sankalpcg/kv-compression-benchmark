r"""
Read one or more kv_bench_results.jsonl files and emit the LaTeX tables for
the synthetic needle sweep.

    python make_table.py results.jsonl [more.jsonl ...]

Files are merged and de-duplicated on (condition, method, ratio, depth, trial),
LAST FILE WINS. The sampler is seeded from (depth, trial), so the same key names
the same sample in every run, and a corrected rerun can therefore be layered over
an earlier sweep by passing it last. That is how the BUZZ rerun replaces the
broken BUZZ rows in the main sweep without redoing the 800 valid runs beside them.

Produces four tables in sn-jnl style (\toprule / \midrule / \botrule):
  1. tab:bench            accuracy against retained-cache fraction, per method
  2. tab:benchdepthloose  accuracy against needle depth at the loosest budget
  3. tab:benchdepth       accuracy against needle depth at the tightest budget
  4. tab:benchcond        query visibility and cross-query reuse, normalised
plus a plain-text summary for sanity-checking before you paste anything.
"""
import sys, json, collections

PATHS = sys.argv[1:] or ["kv_bench_results.jsonl"]

# Order here is the order of rows in every table: the four families in the
# taxonomy's own sequence -- position, attention, geometry, value, quantisation.
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

# load+merge
merged = {}
for path in PATHS:
    n_before = len(merged)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r.setdefault("condition", "qlast")
            key = (r["condition"], r["method"], r["ratio"], r["depth"], r["trial"])
            merged[key] = r
    print("%-44s %5d rows, %+5d new"
          % (path, len(merged) - n_before if n_before == 0 else len(merged),
             len(merged) - n_before), file=sys.stderr)

rows = list(merged.values())
if not rows:
    sys.exit("no results in %s" % ", ".join(PATHS))

model = rows[0].get("model", "?").split("/")[-1]
ctx = rows[0].get("ctx", "?")
conditions = sorted({r["condition"] for r in rows})
ratios = sorted({r["ratio"] for r in rows if r["method"] != "full"}, reverse=True)
depths = sorted({r["depth"] for r in rows})
methods = [m for m in NICE if any(r["method"] == m for r in rows)]

# The headline tables describe the main sweep, which is the qlast condition.
# qfirst and reuse are separate experiments and belong in their own table; mixing
# them into the budget table would average three different questions together.
MAIN = "qlast"


def acc(condition=MAIN, **sel):
    sub = [r for r in rows
           if r["condition"] == condition
           and all(r[k] == v for k, v in sel.items())]
    return (100.0 * sum(r["hit"] for r in sub) / len(sub), len(sub)) if sub else (None, 0)


full_acc, full_n = acc(method="full")

# text summary
print("=" * 72)
print("model=%s  ctx=%s  runs=%d  conditions=%s"
      % (model, ctx, len(rows), ",".join(conditions)))
print("full cache (%s): %.1f%% over %d runs" % (MAIN, full_acc, full_n)
      if full_acc is not None else "full cache: no runs")
print("=" * 72)
hdr = "%-22s" % "method" + "".join("%9s" % ("r=%.2f" % r) for r in ratios)
print(hdr)
print("-" * len(hdr))
for m in methods:
    if m == "full":
        continue
    line = "%-22s" % NICE[m]
    for r in ratios:
        a, n = acc(method=m, ratio=r)
        line += "%9s" % ("%.1f" % a if a is not None else "-")
    print(line)

# table 1
print("\n")
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{Needle retrieval accuracy (\%%) against the fraction of the "
      r"prefill cache retained, measured under one controlled setting: %s at "
      r"an %s-token context, greedy decoding, scores averaged over heads, "
      r"compression applied once after prefill. Each cell is %d runs over five "
      r"needle depths and five trials. The document carries four decoy facts of "
      r"the same surface form as the target, so the answer-bearing span is not "
      r"the only distinctive content. The full-cache row is the uncompressed "
      r"baseline. These numbers are generated here rather than quoted from the "
      r"surveyed papers, for the reason given in "
      r"Section~\ref{subsec:incomparable}.}\label{tab:bench}"
      % (model.replace("_", r"\_"), "{:,}".format(ctx) if isinstance(ctx, int) else ctx,
         max(1, acc(method=methods[-1], ratio=ratios[0])[1])))
print(r"\footnotesize")
print(r"\begin{tabular}{@{}l" + "c" * len(ratios) + r"@{}}")
print(r"\toprule")
print("Method & " + " & ".join("%d\\%%" % round(r * 100) for r in ratios) + r" \\")
print(r"\midrule")
if full_acc is not None:
    print(r"Full cache (no compression) & \multicolumn{%d}{c}{%.1f} \\"
          % (len(ratios), full_acc))
    print(r"\midrule")
for m in methods:
    if m == "full":
        continue
    cells = []
    for r in ratios:
        a, _ = acc(method=m, ratio=r)
        cells.append("%.1f" % a if a is not None else "--")
    print("%s %s & %s \\\\" % (NICE[m], CITE[m], " & ".join(cells)))
print(r"\botrule")
print(r"\end{tabular}")
print(r"\end{table}")

# The depth profile is emitted at both the loosest and the tightest budget.
# At the loosest the policies still differ, so the profile shows where each
# signal spends its budget; at the tightest it shows which ones survive.
CAPTIONS = {
    "loose": (r"Needle retrieval accuracy (\%%) by the position of the "
              r"answer-bearing span, at the loosest budget tested (%d\%% of the "
              r"cache retained). Depth 0 places the needle at the start of the "
              r"document and 1 at the end. At this budget the policies have not "
              r"yet collapsed onto the recency window, so the row shape shows "
              r"where each importance signal actually spends its "
              r"budget.}\label{tab:benchdepthloose}"),
    "tight": (r"Needle retrieval accuracy (\%%) by the position of the "
              r"answer-bearing span, at the tightest budget tested (%d\%% of the "
              r"cache retained). Columns as in Table~\ref{tab:benchdepthloose}. "
              r"This is the measurement that bears on "
              r"Section~\ref{subsec:factual}: a method can hold its average "
              r"accuracy while losing spans at particular "
              r"positions.}\label{tab:benchdepth}"),
}

for which, rt in (("loose", max(ratios) if ratios else None),
                  ("tight", min(ratios) if ratios else None)):
    if rt is None:
        continue
    print("\n")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{" + CAPTIONS[which] % round(rt * 100))
    print(r"\footnotesize")
    print(r"\begin{tabular}{@{}l" + "c" * len(depths) + r"@{}}")
    print(r"\toprule")
    print("Method & " + " & ".join("%.2f" % d for d in depths) + r" \\")
    print(r"\midrule")
    for m in methods:
        if m == "full":
            continue
        cells = []
        for d in depths:
            a, _ = acc(method=m, ratio=rt, depth=d)
            cells.append("%.1f" % a if a is not None else "--")
        print("%s & %s \\\\" % (NICE[m], " & ".join(cells)))
    print(r"\botrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

# Each condition has its own full-cache baseline. qfirst's is not 100%, so
# comparing raw accuracy across conditions charges the policy for a penalty
# the baseline already pays. Normalise first.
COND_RATIOS = [r for r in ratios
               if all(acc(condition=c, method="snapkv", ratio=r)[1] for c in conditions)]
if len(conditions) > 1 and COND_RATIOS:
    base = {}
    for c in conditions:
        a, _ = acc(condition=c, method="full")
        base[c] = a
    print("\n")
    print("%% full-cache baselines by condition: %s"
          % ", ".join("%s %.1f" % (c, base[c]) for c in conditions if base[c] is not None))
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{The effect of query visibility and of cross-query reuse, as "
          r"a percentage of each condition's own uncompressed baseline. "
          r"\emph{Query last} is the main sweep, with the question after the "
          r"document. \emph{Query first} moves the question ahead of the "
          r"document, so every policy can see it while compressing. "
          r"\emph{Reused cache} compresses while the prompt asks about one fact "
          r"and then asks about a different one, which is the cost a "
          r"query-conditioned cache pays when it is reused. Each condition is "
          r"normalised by its own full-cache row because that row is not 100\% "
          r"in every condition.}\label{tab:benchcond}")
    print(r"\footnotesize")
    print(r"\begin{tabular}{@{}l" + "c" * (3 * len(COND_RATIOS)) + r"@{}}")
    print(r"\toprule")
    order = [c for c in ("qlast", "qfirst", "reuse") if c in conditions]
    LABEL = {"qlast": "Query last", "qfirst": "Query first", "reuse": "Reused cache"}
    print("& " + " & ".join(r"\multicolumn{%d}{c}{%s}" % (len(COND_RATIOS), LABEL[c])
                            for c in order) + r" \\")
    print("Method & " + " & ".join("%d\\%%" % round(r * 100)
                                   for _ in order for r in COND_RATIOS) + r" \\")
    print(r"\midrule")
    for m in methods:
        if m == "full":
            continue
        cells = []
        for c in order:
            for r in COND_RATIOS:
                a, _ = acc(condition=c, method=m, ratio=r)
                cells.append("%.1f" % (a * 100.0 / base[c])
                             if a is not None and base.get(c) else "--")
        if all(x == "--" for x in cells):
            continue
        print("%s & %s \\\\" % (NICE[m], " & ".join(cells)))
    print(r"\botrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
