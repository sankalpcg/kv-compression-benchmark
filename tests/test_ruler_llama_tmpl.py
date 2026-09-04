import os
"""Does the sentinel prompt split survive Llama-3.1's chat template?

The split in ruler_prompt() renders the template around two sentinels and cuts
there. That is template-agnostic BY CONSTRUCTION, but Llama-3.1 differs from
Qwen in ways that could still break it: it injects a system block carrying
"Cutting Knowledge Date"/"Today Date" that Qwen has no equivalent of, and it
uses <|begin_of_text|> plus <|start_header_id|> markers rather than <|im_start|>.

Uses NousResearch's ungated mirror, whose tokenizer_config.json carries the same
chat_template as meta-llama's gated repo, so this runs with no token and no
licence. Only the TEMPLATE is under test here, not the weights.
"""
import os, sys

os.environ.update(
    TASKSET="ruler", RULER_CONFIG="4096", DEVICE="cpu",
    MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct",
    RULER_SAMPLES="2", RULER_TRUNC="0",
    RULER_TASKS="niah_single_2,niah_multikey_1,niah_multivalue,vt,fwe,qa_1",
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kv_bench as K
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(K.MODEL)
print("tokenizer:", K.MODEL)
samples = K.ruler_load(tok)
print("loaded %d samples" % len(samples))

for s in samples[::2]:
    pre, suf = K.ruler_prompt(tok, s, "qhidden")
    pre_v, suf_v = K.ruler_prompt(tok, s, "qvisible")
    assert pre + suf == pre_v + suf_v, "split point changes the prompt!"
    assert s["question"] not in pre, "question leaked into the compressed region"
    assert s["question"] in pre_v, "qvisible did not include the question"
    assert s["context"] in pre, "context missing from the compressed region"
    n_pre = len(tok(pre, add_special_tokens=False)["input_ids"])
    n_suf = len(tok(suf, add_special_tokens=False)["input_ids"])
    print("  %-16s ctx+tmpl=%5d tok  suffix=%3d tok  max_new=%d"
          % (s["task"], n_pre, n_suf, s["max_new"]))

s = samples[0]
pre, suf = K.ruler_prompt(tok, s, "qhidden")
print("\n--- boundary under the Llama template ---")
print("PREFIX head :", repr(pre[:160]))
print("PREFIX tail :", repr(pre[-80:]))
print("SUFFIX      :", repr(suf))

# The generation prompt must be the LAST thing before the model speaks, and the
# answer prefix must sit after it -- if the template put the assistant header
# anywhere else the model would be completing the user turn instead.
assert "<|start_header_id|>assistant<|end_header_id|>" in suf, \
    "assistant turn marker is not in the suffix -- model would continue the user turn"
assert suf.rstrip().endswith(s["answer_prefix"].rstrip()), \
    "answer prefix is not last in the suffix"
print("\nLlama template OK: split holds, assistant marker and answer prefix in place.")
