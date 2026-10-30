r"""Paired cluster bootstrap over the RULER sweep.

    python bootstrap_ci.py kaggle_out_ruler/kv_bench_results.jsonl [more ...]

Why a bootstrap at all. The sweep draws ten samples per task, so a task mean
moves in steps of ten points for the hit-scored QA tasks, and an aggregate
built from 13 such means carries real sampling error. Reporting 76.2 against
70.2 without an interval invites the reader to treat a difference as a result
when it may be one sample. This script attaches an interval to every aggregate
the subsection quotes, and a paired interval to every comparison it makes.

Why PAIRED, and why clustered. ruler_load() takes the first RULER_SAMPLES rows
of each task in dataset order, so trial N names the SAME underlying document
and needle in every run -- across methods, ratios, conditions and models. Two
cells therefore differ only in the policy applied to identical inputs, and the
bootstrap must respect that: one draw of trial indices per replicate is shared
by every cell, so the difference distribution reflects the policy and not the
luck of two independent draws. Resampling each cell independently would
inflate the variance of every difference and understate every real effect.

The cluster is the task. A replicate redraws trials WITHIN each task rather
than redrawing tasks, because the macro average fixes the 13 tasks by
construction -- they are the estimand, not a sample from a population of tasks.
Redrawing tasks would answer a different question ("what if RULER had other
tasks") and would widen every interval for a reason nothing here claims.

Reported for each cell: the macro mean and a 95% percentile interval.
Reported for each contrast: the paired difference, its 95% percentile
interval, and the two-sided bootstrap p-value computed as 2*min(P(d<=0),
P(d>=0)), floored at 1/B. An interval that excludes zero is reported as a
separating difference; one that contains it is reported as not separating,
which for the query-visibility null claims is the result being asserted.
"""
import sys, json, collections, re
import numpy as np

PATHS = sys.argv[1:] or ["kaggle_out_ruler/kv_bench_results.jsonl"]
B = 10000
SEED = 20261111
TIGHT = 0.1

TASK_ORDER = ["niah_single_1", "niah_single_2", "niah_single_3",
              "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
              "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
              "qa_1", "qa_2"]
NICE = {"full": "Full cache", "streamingllm": "StreamingLLM", "h2o": "H2O",
        "snapkv": "SnapKV", "pyramidkv": "PyramidKV", "tova": "TOVA",
        "keydiff": "KeyDiff", "valueaware": "Value-aware", "buzz": "BUZZ",
        "kivi": "KIVI (quantisation)"}
ORDER = list(NICE)
EVICTION = ["streamingllm", "h2o", "snapkv", "pyramidkv", "tova", "keydiff",
            "valueaware", "buzz"]

# load
merged = {}
for path in PATHS:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # The taskset is read off the data rather than hardcoded. It stays in the cell
            # key because trial N on LooGLE is a different document from trial N on RULER.
            merged[(r.get("taskset", ""), r.get("model", ""),
                    str(r.get("ruler_config", "")),
                    r["condition"], r["task"],
                    r["method"], r["ratio"], r["trial"])] = r
rows = list(merged.values())
if not rows:
    sys.exit("no benchmark rows found")

TASKSETS = sorted({r.get("taskset", "") for r in rows})
if len(TASKSETS) > 1:
    sys.exit("refusing to mix tasksets in one bootstrap: %s -- the bootstrap "
             "is paired on the trial index, which names the same document only "
             "within one taskset." % ", ".join(TASKSETS))
TASKSET = TASKSETS[0] or "ruler"
BENCH = {"ruler": "RULER", "loogle": "LooGLE"}.get(TASKSET, TASKSET.upper())

MODELS = sorted({r.get("model", "") for r in rows})
CONDS = sorted({r["condition"] for r in rows})
# RULER's 13 tasks in canonical order, then any task the file carries that
# TASK_ORDER does not name. Filtering on TASK_ORDER alone left this empty on a
# LooGLE file and the bootstrap ran over an all-NaN array instead of refusing.
_present = {r["task"] for r in rows}
TASKS = ([t for t in TASK_ORDER if t in _present]
         + sorted(_present - set(TASK_ORDER)))
RATIOS = sorted({r["ratio"] for r in rows if r["method"] != "full"}, reverse=True)
TRIALS = sorted({r["trial"] for r in rows})
N = len(TRIALS)
tix = {t: i for i, t in enumerate(TASKS)}
nix = {n: i for i, n in enumerate(TRIALS)}

# One RULER context length per invocation. The cell key carries no length, and
# the bootstrap is paired, so trial N at 8,192 tokens and trial N at 16,384 are
# unrelated samples. Compare lengths by running this once per length.
CONFIG = sorted({str(r.get("ruler_config", "")) for r in rows})
if len(CONFIG) > 1:
    sys.exit("refusing to mix RULER context lengths in one bootstrap: %s\n"
             "The paired bootstrap shares one draw of trial indices across "
             "cells, but RULER regenerates its documents per configuration, so "
             "trial N at one length is not trial N at another. Run one length "
             "at a time." % ", ".join(CONFIG))


def pretty_len(c):
    """'8192' -> '8{,}192 tokens', for a caption."""
    try:
        return "{:,}".format(int(c)).replace(",", "{,}") + " tokens"
    except (TypeError, ValueError):
        return "the sweep's context length"


def model_slug(name):
    """'Meta-Llama-3.1-8B-Instruct' -> 'llama318b', for a \\label.

    The label used to be tab:<taskset>ci<length>, which collides the moment a
    second model is measured at a length the first one already covers: the
    Qwen 8K table and the Llama 8K table both claimed \\label{tab:rulerci8k},
    LaTeX kept whichever came last, and every \\ref to the other one silently
    pointed at the wrong table. check_tables.py catches the duplicate; this
    stops it being produced.
    """
    t = name.split("/")[-1]
    for drop in ("Meta-", "-Instruct", "-instruct"):
        t = t.replace(drop, "")
    return re.sub(r"[^a-z0-9]", "", t.lower())


CTX = pretty_len(CONFIG[0] if CONFIG else None)
LSUF = ("%dk" % (int(CONFIG[0]) // 1024)) if CONFIG and CONFIG[0].isdigit() else ""
# Every model in the file, not just the primary one: a two-model table that
# named only its primary would be as misleading as no model at all.
MSLUG = "x".join(model_slug(m) for m in MODELS if m)
LABEL_SUF = "-".join(x for x in (MSLUG, LSUF) if x)

# dense array: cells x tasks x trials
by_cell = collections.defaultdict(lambda: np.full((len(TASKS), N), np.nan))
for r in rows:
    c = (r.get("model", ""), r["condition"], r["method"],
         1.0 if r["method"] == "full" else r["ratio"])
    if r["task"] in tix:
        by_cell[c][tix[r["task"]], nix[r["trial"]]] = r["score"]

cells, arr = [], []
partial = []
for c, a in sorted(by_cell.items()):
    if np.isnan(a).any():
        partial.append((c, int(np.isnan(a).sum())))
        continue
    cells.append(c)
    arr.append(a)
arr = np.asarray(arr, dtype=np.float32)
cidx = {c: i for i, c in enumerate(cells)}
print("# cells: %d complete, %d partial (excluded)" % (len(cells), len(partial)),
      file=sys.stderr)
for c, n in partial:
    print("#   partial: %s missing %d" % (" ".join(map(str, c)), n), file=sys.stderr)

# bootstrap
rng = np.random.default_rng(SEED)
point = 100.0 * arr.mean(axis=2).mean(axis=1)          # macro over tasks
boot = np.empty((B, len(cells)), dtype=np.float32)
CHUNK = 200
for s in range(0, B, CHUNK):
    k = min(CHUNK, B - s)
    draws = rng.integers(0, N, size=(k, len(TASKS), N))         # per task, per rep
    g = np.take_along_axis(arr[None, :, :, :], draws[:, None, :, :], axis=3)
    boot[s:s + k] = 100.0 * g.mean(axis=3).mean(axis=2)
print("# bootstrap: %d replicates, seed %d, cluster = task, unit = trial"
      % (B, SEED), file=sys.stderr)


def ci(c):
    i = cidx[c]
    lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
    return point[i], float(lo), float(hi)


def contrast(a, b):
    """Paired difference a - b over shared replicates."""
    d = boot[:, cidx[a]] - boot[:, cidx[b]]
    lo, hi = np.percentile(d, [2.5, 97.5])
    # Clamped to 1: when the paired difference is identically zero in every
    # replicate both tail probabilities are 1, and the doubled minimum gives p = 2.
    p = min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean()))
    return (point[cidx[a]] - point[cidx[b]], float(lo), float(hi),
            max(p, 1.0 / B), bool(lo > 0 or hi < 0))


def have(c):
    return c in cidx


BASE_COND = "qhidden" if "qhidden" in CONDS else CONDS[0]
out, rep = [], []
W = "=" * 78


def say(s=""):
    rep.append(s)


# 1. aggregate intervals
say(W)
say("AGGREGATE MACRO ACCURACY WITH 95%% BOOTSTRAP CI   (condition: %s)" % BASE_COND)
say(W)
for mo in MODELS:
    say("\n%s" % mo.split("/")[-1])
    for m in ORDER:
        line = "  %-22s" % NICE[m]
        for rt in ([1.0] if m == "full" else RATIOS):
            c = (mo, BASE_COND, m, rt)
            if not have(c):
                continue
            v, lo, hi = ci(c)
            line += "  %5.1f [%5.1f,%5.1f]" % (v, lo, hi)
        say(line)

# 2. the contrasts reported in the write-up
say("\n" + W)
say("PAIRED CONTRASTS")
say(W)


def block(title, pairs):
    say("\n### %s" % title)
    for label, a, b in pairs:
        if not (have(a) and have(b)):
            say("  %-54s  [cell missing]" % label)
            continue
        d, lo, hi, p, sep = contrast(a, b)
        say("  %-54s %+6.1f [%+6.1f,%+6.1f] p=%.4f %s"
            % (label, d, lo, hi, p, "SEPARATES" if sep else "not separated"))


for mo in MODELS:
    sm = mo.split("/")[-1]
    # KeyDiff against the best OTHER eviction policy at each budget
    pairs = []
    for rt in RATIOS:
        others = [(point[cidx[(mo, BASE_COND, m, rt)]], m) for m in EVICTION
                  if m != "keydiff" and have((mo, BASE_COND, m, rt))]
        if not others:
            continue
        best = max(others)[1]
        pairs.append(("KeyDiff - %s (best other eviction) @ %g%%"
                      % (NICE[best], 100 * rt),
                      (mo, BASE_COND, "keydiff", rt), (mo, BASE_COND, best, rt)))
    block("%s: does KeyDiff lead the eviction policies?" % sm, pairs)

    # BUZZ against the second-worst
    pairs = []
    for rt in RATIOS:
        others = sorted((point[cidx[(mo, BASE_COND, m, rt)]], m) for m in EVICTION
                        if m != "buzz" and have((mo, BASE_COND, m, rt)))
        if not others:
            continue
        worst = others[0][1]
        pairs.append(("%s (2nd worst) - BUZZ @ %g%%" % (NICE[worst], 100 * rt),
                      (mo, BASE_COND, worst, rt), (mo, BASE_COND, "buzz", rt)))
    block("%s: is BUZZ last?" % sm, pairs)

    block("%s: quantisation against the best eviction policy" % sm,
          [("KIVI - KeyDiff @ %g%%" % (100 * rt),
            (mo, BASE_COND, "kivi", rt), (mo, BASE_COND, "keydiff", rt))
           for rt in RATIOS])

    block("%s: SnapKV against PyramidKV" % sm,
          [("SnapKV - PyramidKV @ %g%%" % (100 * rt),
            (mo, BASE_COND, "snapkv", rt), (mo, BASE_COND, "pyramidkv", rt))
           for rt in RATIOS])

# query visibility, every policy, every budget, every model
for mo in MODELS:
    if not ("qvisible" in CONDS and "qhidden" in CONDS):
        break
    block("%s: query visible minus query hidden" % mo.split("/")[-1],
          [("%s @ %g%%" % (NICE[m], 100 * rt),
            (mo, "qvisible", m, rt), (mo, "qhidden", m, rt))
           for m in ORDER if m != "full" for rt in RATIOS])
    say("  %-54s %+6.1f %s"
        % ("Full cache (control) @ 100%",
           point[cidx[(mo, "qvisible", "full", 1.0)]]
           - point[cidx[(mo, "qhidden", "full", 1.0)]]
           if have((mo, "qvisible", "full", 1.0)) else float("nan"),
           "<- uncompressed baseline, reordering effect only"))

# cross-model retention, paired on the same samples
if len(MODELS) > 1:
    a_m, b_m = MODELS[0], MODELS[1]
    block("retention is model-specific: %s - %s"
          % (a_m.split("/")[-1], b_m.split("/")[-1]),
          [("%s @ %g%%" % (NICE[m], 100 * rt),
            (a_m, BASE_COND, m, rt), (b_m, BASE_COND, m, rt))
           for m in ORDER if m != "full" for rt in RATIOS])

# LaTeX
PRIMARY = max(MODELS, key=lambda m: (
    sum(1 for c in cells if c[0] == m),
    point[cidx[(m, BASE_COND, "full", 1.0)]] if have((m, BASE_COND, "full", 1.0)) else 0))
short = PRIMARY.split("/")[-1]
out.append(r"\begin{table}[t]")
out.append(r"\centering")
# The condition is read off the data: BASE_COND falls back to whatever single
# condition a qvisible-only file carries. The label carries the length so two
# lengths in one manuscript do not share a label.
WHERE = ("the question held outside the compressed region"
         if BASE_COND == "qhidden" else
         "the question inside the compressed region, visible to the policy")
out.append(r"\caption{Macro accuracy on %s at %s with 95 per cent bootstrap "
           r"confidence intervals, for %s with %s. Intervals are percentile "
           r"intervals from %s replicates of a paired cluster bootstrap that "
           r"resamples the trials within each task and shares one draw "
           r"across every policy, so the intervals on differences in the text "
           r"are narrower than a reader would obtain by overlapping the "
           r"intervals below."
           r"}\label{tab:%sci-%s}"
           % (BENCH, CTX, short, WHERE,
              "{:,}".format(B).replace(",", "{,}"), TASKSET, LABEL_SUF))
out.append(r"\footnotesize")
# Column count from the data. A hardcoded "lccc" renders two empty columns
# on a single-budget file and raises "Extra alignment tab" on a four-budget
# one; the \multicolumn below already counts len(RATIOS).
out.append(r"\begin{tabular}{@{}l%s@{}}" % ("c" * len(RATIOS)))
out.append(r"\toprule")
out.append("Method & " + " & ".join(r"%g\%%" % (100 * rt) for rt in RATIOS) + r" \\")
out.append(r"\midrule")
if have((PRIMARY, BASE_COND, "full", 1.0)):
    v, lo, hi = ci((PRIMARY, BASE_COND, "full", 1.0))
    out.append(r"Full cache (no compression) & \multicolumn{%d}{c}{%.1f (%.1f--%.1f)} \\"
               % (len(RATIOS), v, lo, hi))
    out.append(r"\midrule")
for m in ORDER:
    if m == "full":
        continue
    cs = []
    for rt in RATIOS:
        c = (PRIMARY, BASE_COND, m, rt)
        if not have(c):
            cs.append("--")
            continue
        v, lo, hi = ci(c)
        cs.append("%.1f (%.1f--%.1f)" % (v, lo, hi))
    out.append("%s & %s \\\\" % (NICE[m], " & ".join(cs)))
out.append(r"\botrule")
out.append(r"\end{tabular}")
out.append(r"\end{table}")

print("\n".join(out))
print("\n".join(rep), file=sys.stderr)
print("\n# primary model for tab:rulerci: %s" % short, file=sys.stderr)
