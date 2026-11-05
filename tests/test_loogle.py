r"""Check the LooGLE loader, prompt split and F1 scorer.

    python test_loogle.py            # scorer only, no network
    python test_loogle.py --load     # also pull a few documents and split them

The scorer is checked against cases whose answer is not a matter of opinion:
an exact match must be 1.0, a disjoint answer 0.0, and a partial overlap the
harmonic mean of its own precision and recall. The normalisation is checked
separately, because an F1 computed without it silently disagrees with every
published LooGLE number.

The loader is checked for the property the whole naturalistic comparison rests
on: that `ruler_prompt` puts the question OUTSIDE the compressed prefix under
`qhidden` and INSIDE it under `qvisible`, with the same context either way.
"""
import os, sys

os.environ.setdefault("TASKSET", "loogle")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
os.environ.setdefault("LOOGLE_SAMPLES", "3")
os.environ.setdefault("LOOGLE_CTX", "2048")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kv_bench as K

fails = []


def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol
    print("  %-56s %.4f %s" % (name, got, "ok" if ok else "EXPECTED %.4f" % want))
    if not ok:
        fails.append(name)


print("--- scorer ---")
check("exact match", K.loogle_score("t", "Jerez de la Frontera", ["Jerez de la Frontera"]), 1.0)
check("case and article differences only",
      K.loogle_score("t", "the JEREZ de la Frontera", ["Jerez de la Frontera"]), 1.0)
check("trailing punctuation only",
      K.loogle_score("t", "Jerez de la Frontera.", ["Jerez de la Frontera"]), 1.0)
check("no overlap", K.loogle_score("t", "Madrid", ["Jerez de la Frontera"]), 0.0)
# 2 of 3 predicted tokens correct, 2 of 4 gold tokens: P=2/3, R=1/2, F1=0.5714
check("partial overlap", K.loogle_score("t", "born in Jerez", ["he was born in Madrid"]),
      2 * (2 / 3) * (2 / 5) / ((2 / 3) + (2 / 5)))
check("empty prediction", K.loogle_score("t", "", ["something"]), 0.0)
check("best of several references",
      K.loogle_score("t", "Madrid", ["Jerez de la Frontera", "Madrid"]), 1.0)
# A yes/no answer is one token, so the metric is brittle there by construction.
check("one-token answer, right", K.loogle_score("t", "No.", ["No."]), 1.0)
# A correct yes/no answer given in a sentence scores 0.25: seven normalised
# tokens against a one-token reference gives P=1/7, R=1. Token F1 is weak on
# short references, so read the yes/no questions in this set with that in mind.
check("one-token answer, verbose",
      K.loogle_score("t", "No, he did not receive any awards.", ["No."]),
      2 * (1 / 7) * 1.0 / ((1 / 7) + 1.0))

print("\n--- normalisation ---")
print("  %r -> %r" % ("The, Province of Cadiz!", K._norm_answer("The, Province of Cadiz!")))
if K._norm_answer("The, Province of Cadiz!") != "province of cadiz":
    fails.append("normalisation")
    print("  EXPECTED 'province of cadiz'")

if "--load" in sys.argv:
    print("\n--- loader and prompt split ---")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(K.MODEL)
    samples = K.loogle_load(tok)
    print("  %d sample(s), %d distinct document(s)"
          % (len(samples), len({s["doc"] for s in samples})))
    if len({s["doc"] for s in samples}) != len(samples):
        fails.append("one question per document")
        print("  FAIL: documents repeat, so the aggregate would average over "
              "questions about one text")
    n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    for s in samples:
        hid_pre, hid_suf = K.ruler_prompt(tok, s, "qhidden")
        vis_pre, vis_suf = K.ruler_prompt(tok, s, "qvisible")
        q_in_hidden_prefix = s["question"].strip() in hid_pre
        q_in_visible_prefix = s["question"].strip() in vis_pre
        print("  doc %-28s prefix %5d/%5d tok  question in prefix: "
              "qhidden=%s qvisible=%s  max_new=%d"
              % (str(s["doc"])[:28], n(hid_pre), n(vis_pre),
                 q_in_hidden_prefix, q_in_visible_prefix, s["max_new"]))
        if q_in_hidden_prefix:
            fails.append("question leaked into the compressed prefix under qhidden")
        if not q_in_visible_prefix:
            fails.append("question missing from the compressed prefix under qvisible")
        if hid_pre + hid_suf != vis_pre + vis_suf:
            fails.append("the two conditions do not render the same full prompt")
        if K.LOOGLE_CTX and n(hid_pre) > K.LOOGLE_CTX + 64:
            fails.append("context not truncated to LOOGLE_CTX")

print()
if fails:
    for f in sorted(set(fails)):
        print("FAIL " + f)
    sys.exit(1)
print("loogle OK")
