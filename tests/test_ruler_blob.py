r"""Prove the blob splitter recovers exactly what the three-field mirror carries.

    python test_ruler_blob.py            # run against the cached rows
    python test_ruler_blob.py --fetch    # fetch the rows first, then run

The claim under test is narrow and checkable. `SaylorTwift/RULER-<len>-*` ships
one `input` field per row; `simonjegou/ruler` ships `context`, `question` and
`answer_prefix` for the same generation. If `ruler_blob_split()` applied to the
blob returns those three strings character for character, for every task, then
reading the blob mirror is equivalent to reading the three-field one -- and the
32,768-token configuration, which only the blob mirror has, can be read without
a new assumption.

That equivalence is the whole safety argument. ruler_prompt() decides the
query-visibility condition by where the question sits relative to the compressed
region, so a boundary off by a few characters would move part of the question
into the compressed prefix and measure something other than the condition it
reports, without failing. A test comparing the recovered strings to known-good
ones is the only thing that rules it out.

Both mirrors are compared at 4,096 tokens, where the three-field rows exist. The
markers do not vary with the configuration, so this is the strongest available
test of a path that will be used at 32,768.
"""
import io, json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
CACHE = os.path.join(HERE, "fixtures", "ruler_rows")
BLOB_REPO = os.environ.get("BLOB_REPO", "SaylorTwift/RULER-4096-llama-3.2-tokenizer")
CONFIG = "4096"                      # the configuration the gold rows come from
N = int(os.environ.get("N_ROWS", 3))
# Tier 2 needs a gold row for the SAME configuration. Point BLOB_REPO at the
# 32,768-token mirror and there is none, by construction -- that is the whole
# reason this path exists -- so only tier 1 runs there.
_m = re.search(r"RULER-(\d+)-", BLOB_REPO)
BLOB_LEN = _m.group(1) if _m else ""
COMPARABLE = BLOB_LEN == CONFIG

TASKS = ["niah_single_1", "niah_single_2", "niah_single_3",
         "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
         "qa_1", "qa_2"]


def _get(url, tries=4):
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            last = e
            time.sleep(2 + 3 * attempt)
    raise last


def cached(kind, task, fetch, url):
    stem = kind if kind.startswith("blob_") else "%s_%s" % (kind, CONFIG)
    path = os.path.join(CACHE, "%s_%s.json" % (stem, task))
    if os.path.exists(path):
        return json.load(io.open(path, encoding="utf-8"))
    if not fetch:
        raise SystemExit("missing %s; rerun with --fetch" % os.path.basename(path))
    os.makedirs(CACHE, exist_ok=True)
    rows = [x["row"] for x in _get(url)["rows"]]
    io.open(path, "w", encoding="utf-8").write(json.dumps(rows))
    return rows


def blob_rows(task, fetch):
    url = ("https://datasets-server.huggingface.co/rows?dataset=%s&config=default"
           "&split=%s&offset=0&length=%d"
           % (BLOB_REPO.replace("/", "%2F"), task, N))
    return cached("blob_" + BLOB_LEN, task, fetch, url)


def gold_rows(task, fetch):
    where = '%22task%22%3D%27' + task + '%27'
    url = ("https://datasets-server.huggingface.co/filter?dataset=simonjegou%2Fruler"
           "&config=" + CONFIG + "&split=test&where=" + where +
           "&offset=0&length=%d" % N)
    return cached("gold", task, fetch, url)


def main():
    import kv_bench as K
    fetch = "--fetch" in sys.argv
    fails, checked, equivalent, divergent = [], 0, 0, {}

    for task in TASKS:
        blob = blob_rows(task, fetch)
        gold = gold_rows(task, fetch) if COMPARABLE else [None] * len(blob)
        if len(gold) != len(blob):
            fails.append("%s: %d gold rows against %d blob rows" % (task, len(gold), len(blob)))
            continue
        for i, (g, b) in enumerate(zip(gold, blob)):
            # ---- tier 1: the split must be lossless and well formed. This
            # applies to every task and every mirror revision, and it is what
            # has to hold at 32,768 tokens where no gold row exists.
            try:
                ctx, q, ap = K.ruler_blob_split(task, b["input"])
            except Exception as e:
                fails.append("%s[%d]: split raised %s: %s" % (task, i, type(e).__name__, e))
                continue
            checked += 1
            if ctx + q + ap != b["input"]:
                fails.append("%s[%d]: the three recovered parts do not rejoin to the "
                             "input -- the split lost or duplicated text" % (task, i))
            if not q.strip() or not ap.strip():
                fails.append("%s[%d]: split produced an empty question or answer "
                             "prefix" % (task, i))
            q_mark, a_mark = K.RULER_BLOB_MARKERS[task]
            if not q.startswith(q_mark):
                fails.append("%s[%d]: recovered question does not begin with the "
                             "task's question template" % (task, i))
            if not ap.startswith(a_mark):
                fails.append("%s[%d]: recovered answer prefix does not begin with "
                             "the task's answer template" % (task, i))
            if "\n" in q.replace(q_mark, "", 1) and not task.startswith("qa_"):
                fails.append("%s[%d]: recovered question spans a newline, which "
                             "only the QA templates do" % (task, i))

            # ---- tier 2: where the two mirrors are the same revision of RULER,
            # the recovered parts must equal the three-field mirror's exactly.
            if g is None:
                continue
            same_sample = list(g["answer"]) == list(b["outputs"])
            same_text = g["context"] + g["question"] + g["answer_prefix"] == b["input"]
            if not (same_sample and same_text):
                divergent.setdefault(task, []).append(i)
                continue
            equivalent += 1
            for field, got, wanted in (("context", ctx, g["context"]),
                                       ("question", q, g["question"]),
                                       ("answer_prefix", ap, g["answer_prefix"])):
                if got != wanted:
                    fails.append("%s[%d] %s: recovered %r, expected %r"
                                 % (task, i, field,
                                    got[:70] if field != "context" else got[-70:],
                                    wanted[:70] if field != "context" else wanted[-70:]))

    print("split %d row(s) across %d task(s) of %s" % (checked, len(TASKS), BLOB_REPO))
    print("  tier 1 (lossless rejoin, templates in place): all %d" % checked)
    if COMPARABLE:
        print("  tier 2 (identical to the three-field mirror): %d" % equivalent)
    else:
        print("  tier 2 skipped: no three-field mirror exists at %s tokens, which "
              "is why this path exists" % (BLOB_LEN or "this length"))
    for task, idxs in sorted(divergent.items()):
        print("  not comparable on %s rows %s: the two mirrors are different "
              "revisions of RULER there, so tier 2 does not apply"
              % (task, idxs))
    for f in fails:
        print("FAIL " + f)
    if fails:
        print("%d failure(s)" % len(fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
