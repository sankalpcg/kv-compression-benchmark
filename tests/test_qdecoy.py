
"""Local check of the qdecoy condition. No GPU, no model, no download."""
import os, sys
os.environ["RULER_CONDITIONS"] = "qhidden,qvisible,qdecoy"
os.environ["DEVICE"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kv_bench as kb


class StubTok(object):
    """Whitespace tokeniser. apply_chat_template raises so ruler_prompt uses
    its documented fallback, which is the path a model without a template
    takes anyway."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def apply_chat_template(self, *a, **k):
        raise RuntimeError("no template")


def mk(task, i, q):
    return dict(task=task, index=i, context="CTX-%s-%d" % (task, i),
                question=q, answer_prefix="Answer:", answers=["x"], max_new=8)


tok = StubTok()
samples = [
    mk("qa_1", 0, "one two three four five"),      # 5
    mk("qa_1", 1, "alpha beta"),                   # 2
    mk("qa_1", 2, "a b c d"),                      # 4
    mk("vt", 0, "p q r"),                          # 3
    mk("vt", 1, "s t u"),                          # 3
    mk("solo", 0, "lonely question here"),
]
kb.attach_decoys(tok, samples)

fail = []

def check(cond, got, want):
    if got != want:
        fail.append("%s\n  got  %r\n  want %r" % (cond, got, want))

# closest length wins, from the same task, never itself
check("qa_1#0 decoy", samples[0]["decoy"], "a b c d")          # 5 -> 4 beats 2
check("qa_1#1 decoy", samples[1]["decoy"], "a b c d")          # 2 -> 4 beats 5
check("qa_1#2 decoy", samples[2]["decoy"], "one two three four five")  # 4 -> 5 beats 2
check("vt#0 decoy", samples[3]["decoy"], "s t u")
check("vt#1 decoy", samples[4]["decoy"], "p q r")
if "decoy" in samples[5]:
    fail.append("single-sample task should get no decoy")
for s in samples[:5]:
    if s["decoy"] == s["question"]:
        fail.append("task %s index %s borrowed its own question" % (s["task"], s["index"]))

# the three splits
s0 = samples[0]
pre_h, suf_h = kb.ruler_prompt(tok, s0, "qhidden")
pre_v, suf_v = kb.ruler_prompt(tok, s0, "qvisible")
pre_d, suf_d = kb.ruler_prompt(tok, s0, "qdecoy")

check("qhidden prefix", pre_h, "CTX-qa_1-0")
check("qhidden suffix", suf_h, "\n\none two three four five\nAnswer:")
check("qvisible prefix", pre_v, "CTX-qa_1-0\n\none two three four five")
check("qvisible suffix", suf_v, "\nAnswer:")
check("qdecoy prefix", pre_d, "CTX-qa_1-0\n\na b c d")
check("qdecoy suffix", suf_d, "\n\none two three four five\nAnswer:")

# what the ablation rests on: qdecoy scores the same position qvisible does,
# and answers the same question qhidden does.
if suf_d != suf_h:
    fail.append("qdecoy must forward exactly what qhidden forwards")
if pre_d.rsplit("\n\n", 1)[0] != pre_v.rsplit("\n\n", 1)[0]:
    fail.append("qdecoy and qvisible must share everything before the question")
if pre_d == pre_v:
    fail.append("qdecoy prefix must not contain the real question")
if s0["question"] in pre_d:
    fail.append("the real question leaked into the scored prefix")

# a missing decoy is an error, not a silent fallback to the real question
try:
    kb.ruler_prompt(tok, samples[5], "qdecoy")
    fail.append("missing decoy should raise")
except RuntimeError:
    pass

# the other two conditions are untouched
plain = mk("qa_1", 9, "untouched question")
check("no-decoy qhidden prefix", kb.ruler_prompt(tok, plain, "qhidden")[0], "CTX-qa_1-9")
check("no-decoy qvisible prefix", kb.ruler_prompt(tok, plain, "qvisible")[0],
      "CTX-qa_1-9\n\nuntouched question")

if fail:
    print("FAIL (%d)" % len(fail))
    for f in fail:
        print(" -", f)
    raise SystemExit(1)
print("ok: decoy selection, the three prompt splits, and the guard")
