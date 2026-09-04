"""Check the RULER loader, prompt split and scorer WITHOUT loading a model."""
import os, sys
os.environ.update(
    TASKSET="ruler", RULER_CONFIG="4096", DEVICE="cpu",
    MODEL="Qwen/Qwen2.5-0.5B-Instruct",
    RULER_SAMPLES="2", RULER_TRUNC="0",
    RULER_TASKS="niah_single_2,niah_multikey_1,niah_multivalue,vt,fwe,qa_1",
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kv_bench as K
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(K.MODEL)
samples = K.ruler_load(tok)
print("loaded %d samples" % len(samples))

for s in samples[::2]:                      # one per task
    pre, suf = K.ruler_prompt(tok, s, "qhidden")
    pre_v, suf_v = K.ruler_prompt(tok, s, "qvisible")
    n_pre = len(tok(pre, add_special_tokens=False)["input_ids"])
    n_suf = len(tok(suf, add_special_tokens=False)["input_ids"])
    n_pre_v = len(tok(pre_v, add_special_tokens=False)["input_ids"])
    # invariants
    assert pre + suf == pre_v + suf_v, "split point changes the prompt!"
    assert s["question"] not in pre, "question leaked into the compressed region"
    assert s["question"] in pre_v, "qvisible did not include the question"
    assert s["context"] in pre
    print("  %-16s ctx+tmpl=%5d tok  suffix=%3d tok  qvis_prefix=%5d  max_new=%d  n_ans=%d"
          % (s["task"], n_pre, n_suf, n_pre_v, s["max_new"], len(s["answers"])))

print("\n--- one full prompt, boundary shown ---")
s = samples[0]
pre, suf = K.ruler_prompt(tok, s, "qhidden")
print("PREFIX tail :", repr(pre[-120:]))
print("SUFFIX      :", repr(suf))
print("ANSWERS     :", s["answers"])

print("\n--- scorer ---")
cases = [
    ("niah_multivalue", "the numbers are 111 and 222", ["111", "222"], 1.0),
    ("niah_multivalue", "the numbers are 111", ["111", "222"], 0.5),
    ("qa_1",            "Paris, probably",  ["Paris", "France"], 1.0),
    ("qa_1",            "Berlin",           ["Paris", "France"], 0.0),
    ("niah_single_2",   "\x00\x01  ANSWER-42 ", ["answer-42"], 1.0),
    ("vt",              "", ["x1"], 0.0),
]
for task, pred, ans, want in cases:
    got = K.ruler_score(task, pred, ans)
    print("  %-16s %-26r -> %.2f  %s" % (task, pred, got,
                                         "ok" if abs(got - want) < 1e-9 else "FAIL"))
    assert abs(got - want) < 1e-9
print("\nplumbing OK")
