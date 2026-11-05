import os
"""
End-to-end RULER smoke test on CPU with a real model.

The point is a KNOWN-GOOD ROW. Contexts are truncated so the CPU can hold the
attention matrices, which usually cuts the needle out, so samples are filtered
down to those whose answers still appear in the truncated context. `full` must
then score 1.0 -- if it does not, the fault is in the harness, not the policy,
which is the invariant that caught all three previous defects in this project.
"""
import os, sys, time
os.environ.update(
    TASKSET="ruler", RULER_CONFIG="4096", DEVICE="cpu",
    MODEL="Qwen/Qwen2.5-0.5B-Instruct",
    CHUNK="128", OBS_WINDOW="32", RECENT="32", SINK="4",
    RULER_SAMPLES="8", RULER_TRUNC="1280", RULER_MAX_NEW="24",
    RULER_TASKS="niah_single_2,niah_multikey_1",
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import kv_bench as K

tok, model = K.build_model()
samples = K.ruler_load(tok)

keep = [s for s in samples
        if all(str(a).lower() in s["context"].lower() for a in s["answers"])]
print("%d/%d samples still contain their answer after truncation"
      % (len(keep), len(samples)))
keep = keep[:2]
assert keep, "truncation removed every needle; raise RULER_TRUNC"

rows = []
for cond in ("qhidden", "qvisible"):
    for method, ratio in [("full", 1.0), ("streamingllm", 0.5),
                          ("snapkv", 0.5), ("pyramidkv", 0.2), ("kivi", 0.5)]:
        for s in keep:
            pre, suf = K.ruler_prompt(tok, s, cond)
            ids = tok(pre, add_special_tokens=False,
                      return_tensors="pt").input_ids
            S = ids.shape[1]
            t0 = time.time()
            cache, sig, _ = K.prefill_with_scores(model, ids)
            cache, kept = K.compress(cache, method, S, ratio, sig)
            cache, logits, S_eff = K.extend_with_text(model, tok, cache, suf, S)
            text = K.decode(model, tok, cache, logits, orig_len=S_eff,
                            max_new=s["max_new"])
            score = K.ruler_score(s["task"], text, s["answers"])
            rows.append((cond, method, ratio, s["task"], score, text.strip()))
            print("  %-9s %-13s r=%.2f %-16s S=%4d kept=%6.1f -> %.2f  %r"
                  % (cond, method, ratio, s["task"], S, kept, score,
                     text.strip()[:60]))

print("\n--- invariants ---")
empty = [r for r in rows if not r[5]]
print("empty outputs: %d/%d" % (len(empty), len(rows)))
full_rows = [r for r in rows if r[1] == "full"]
full_mean = sum(r[4] for r in full_rows) / len(full_rows)
print("full-cache mean score: %.2f over %d runs" % (full_mean, len(full_rows)))
assert not empty, "EMPTY OUTPUT -- prompt/turn-marker defect, do not push"
assert full_mean > 0.4, ("full cache cannot read its own needle (%.2f) -- "
                         "harness defect, do not push" % full_mean)
print("\nE2E OK: baseline reads the needle, no empty outputs.")
