r"""Derive the per-task split markers that let a blob-schema RULER mirror be read.

    python derive_ruler_templates.py            # derive from the cached gold rows
    python derive_ruler_templates.py --fetch    # fetch the gold rows first

Why this exists. `ruler_load()` needs `context`, `question` and `answer_prefix`
as three separate fields, because the query-visibility condition is defined by
where the question sits relative to the compressed region: `ruler_prompt()` puts
the question inside the compressed prefix for `qvisible` and in the uncompressed
suffix for `qhidden`. `simonjegou/ruler`, the mirror the harness loads, carries
the three fields -- and stops at a 16,384-token configuration. The mirrors that
go further (SaylorTwift/RULER-32768-*, -65536-*, -131072-*) carry a single
`input` blob instead, with the task name as the split.

Those mirrors are the same generation: at 16,384 their `niah_single_1` row 0
answers 7700828 for the key `solid-few`, as `simonjegou`'s does, and their
`input` is exactly `context + question + answer_prefix` concatenated. So the
three fields are recoverable, but only if the boundaries are placed by something
better than a guess -- `question + answer_prefix` is newline-free on 11 of the 13
tasks, so "split at the last newline" looks sufficient and is wrong for `qa_1`
and `qa_2`, which carry two newlines inside that span.

What this derives instead. RULER generates each task's question and answer
prefix from a fixed template with slots. Taking the longest common prefix of the
`question` field across several rows of one task recovers the template text that
precedes the first slot, and the same over `answer_prefix` recovers its leading
template text. Both are length-invariant -- the configuration changes the
haystack, not the templates -- so markers derived at 4,096 tokens apply at
32,768. Splitting is then: find the LAST occurrence of the question marker, cut
there; find the LAST occurrence of the answer marker in the remainder, cut there.
The last occurrence rather than the first, so a marker that also appears
somewhere in the haystack cannot move the boundary.

The output is pasted into kv_bench.py, which is the only file build_kernel.py
ships. test_ruler_blob.py then checks the result against the mirror itself.
"""
import io, json, os, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("RULER_ROW_CACHE") or os.path.join(HERE, "_ruler_rows")
GOLD_REPO = "simonjegou/ruler"
GOLD_CONFIG = os.environ.get("GOLD_CONFIG", "4096")

TASKS = ["niah_single_1", "niah_single_2", "niah_single_3",
         "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
         "qa_1", "qa_2"]


def _get(url, tries=5):
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:                      # 500/502 from the rows API
            last = e
            time.sleep(2 + 3 * attempt)
    raise last


def gold_rows(task, n=3, fetch=False):
    """Rows carrying the three separate fields, cached on disk once fetched."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "gold_%s_%s.json" % (GOLD_CONFIG, task))
    if os.path.exists(path):
        return json.load(io.open(path, encoding="utf-8"))
    if not fetch:
        raise SystemExit("no cached rows for %s; rerun with --fetch" % task)
    where = urllib.parse.quote('"task"=' + "'" + task + "'", safe="")
    url = ("https://datasets-server.huggingface.co/filter"
           "?dataset=%s&config=%s&split=test&where=%s&offset=0&length=%d"
           % (urllib.parse.quote(GOLD_REPO, safe=""), GOLD_CONFIG, where, n))
    rows = [x["row"] for x in _get(url)["rows"]]
    io.open(path, "w", encoding="utf-8").write(json.dumps(rows))
    return rows


def common_prefix(strings):
    """Longest common prefix. The template text before the first slot."""
    if not strings:
        return ""
    out = strings[0]
    for s in strings[1:]:
        i = 0
        while i < min(len(out), len(s)) and out[i] == s[i]:
            i += 1
        out = out[:i]
    return out


def trim_into_slot(marker):
    """Cut a marker back out of a slot it has eaten into.

    A common prefix taken over a handful of rows can run PAST the end of the
    template and into the slot, whenever those rows happen to agree on the
    slot's first characters. `qa_2` is the real case: three sampled questions
    all began with "W", so the prefix came out as "...Question: W" and a
    question beginning "Did..." at another configuration would not match it.

    Every RULER template places its slot after a space or a newline, so cutting
    back to the last one removes the eaten characters. This cannot move the
    boundary the marker locates: shortening a prefix leaves its starting
    position unchanged, and only costs specificity, which `rfind` absorbs.
    Applied only when the marker ends inside a word, so the markers that
    legitimately end in punctuation ("...are:") are left alone.
    """
    if not marker or not marker[-1].isalnum():
        return marker
    cut = max(marker.rfind(" "), marker.rfind("\n"))
    return marker[:cut + 1] if cut >= 0 else marker


def blob_rows(task):
    """Cached rows from a blob mirror, used only to cross-check the markers."""
    path = os.path.join(CACHE, "blob_%s_%s.json" % (GOLD_CONFIG, task))
    if not os.path.exists(path):
        return []
    return json.load(io.open(path, encoding="utf-8"))


def shorten_until_present(marker, texts):
    """Cut a marker down until every blob row contains it.

    The two mirrors are not always the same revision of RULER. On `vt` the
    three-field mirror still carries the upstream typo "are assgined the value"
    where the blob mirror carries "are assigned", and it differs in whitespace
    around the instruction as well. A marker taken from one revision alone would
    simply not be found in the other, and ruler_blob_split() would raise on
    every row of that task.

    Shortening is safe for the boundary: the marker is located by its START, and
    a prefix of a prefix starts in the same place. What it costs is specificity,
    which the rfind in ruler_blob_split absorbs.
    """
    if not texts:
        return marker, False
    while marker and not all(marker in x for x in texts):
        marker = marker[:-1]
    return marker, True


def derive(fetch=False):
    markers, maxnew, problems = {}, {}, []
    for task in TASKS:
        rows = gold_rows(task, fetch=fetch)
        if len(rows) < 2:
            problems.append("%s: only %d row(s), a common prefix needs two"
                            % (task, len(rows)))
            continue
        q = trim_into_slot(common_prefix([r["question"] for r in rows]))
        a = trim_into_slot(common_prefix([r["answer_prefix"] for r in rows]))
        # Whatever the trim produced must still be a genuine prefix of every
        # sampled row, or the derivation is not describing these rows at all.
        for r in rows:
            if not r["question"].startswith(q) or not r["answer_prefix"].startswith(a):
                problems.append("%s: derived marker is not a prefix of a sampled row" % task)
                break
        mn = {int(r["max_new_tokens"]) for r in rows}
        # Cross-check against the blob mirror this marker will actually be
        # applied to, and shorten if the two mirrors are different revisions.
        blobs = [r["input"] for r in blob_rows(task)]
        q2, checked_q = shorten_until_present(q, blobs)
        a2, checked_a = shorten_until_present(a, blobs)
        if checked_q and q2 != q:
            problems.append("NOTE %s: question marker shortened for the blob mirror,"
                            " %r -> %r" % (task, q[-40:], q2[-40:]))
        if checked_a and a2 != a:
            problems.append("NOTE %s: answer marker shortened for the blob mirror,"
                            " %r -> %r" % (task, a[-40:], a2[-40:]))
        q, a = trim_into_slot(q2), trim_into_slot(a2)
        if not blobs:
            problems.append("NOTE %s: no cached blob rows, marker unverified "
                            "against the blob mirror" % task)
        markers[task] = (q, a)
        maxnew[task] = sorted(mn)[0]
        if len(mn) > 1:
            problems.append("%s: max_new_tokens varies across rows %s" % (task, sorted(mn)))
        # A marker must actually place a boundary: empty, or absent from the
        # field it is meant to locate, means the derivation failed for this task.
        if not q:
            problems.append("%s: question marker is empty" % task)
        if not a:
            problems.append("%s: answer-prefix marker is empty" % task)
        # And it must not be ambiguous inside the question itself.
        for r in rows:
            tail = r["question"] + r["answer_prefix"]
            if a and tail.count(a) != 1:
                problems.append("%s: answer marker %r occurs %d times in one tail"
                                % (task, a[:40], tail.count(a)))
                break
    return markers, maxnew, problems


def main():
    fetch = "--fetch" in sys.argv
    markers, maxnew, problems = derive(fetch=fetch)
    print("# Derived from %s config %s, %d tasks. Paste into kv_bench.py."
          % (GOLD_REPO, GOLD_CONFIG, len(markers)))
    print("RULER_BLOB_MARKERS = {")
    for task in TASKS:
        if task not in markers:
            continue
        q, a = markers[task]
        print("    %r: (%r,\n     %s%r)," % (task, q, " " * len(repr(task)), a))
    print("}")
    print("RULER_BLOB_MAXNEW = {")
    for task in TASKS:
        if task in maxnew:
            print("    %r: %d," % (task, maxnew[task]))
    print("}")
    # A NOTE records something the derivation handled and the reader should
    # know; only a real PROBLEM fails the run.
    notes = [p for p in problems if p.startswith("NOTE ")]
    hard = [p for p in problems if not p.startswith("NOTE ")]
    for pr in notes:
        print(pr, file=sys.stderr)
    for pr in hard:
        print("PROBLEM: " + pr, file=sys.stderr)
    print("# %d task(s) derived, %d note(s), %d problem(s)"
          % (len(markers), len(notes), len(hard)), file=sys.stderr)
    return 1 if (hard or len(markers) != len(TASKS)) else 0


if __name__ == "__main__":
    sys.exit(main())
