"""KV-cache compression benchmark harness.

Compares cache-eviction policies on a common footing: one model, one benchmark,
one budget definition, one decode loop. It measures accuracy retention against
compression ratio and nothing else. Throughput and latency are hardware-bound
and are not reported.

How it works: run prefill, slice the resulting KV cache, decode manually. No
monkey-patching, so it runs on any HF causal LM and behaves the same on CPU and
GPU. The eviction signals come from a chunked prefill, reduced to column sums as
each chunk is processed so the full [B,H,Q,K] tensor is never held.

Deviations from the original papers, which belong beside any number this
produces:

1. Eviction happens once, after prefill. H2O and TOVA also evict during
   generation in their papers, so they are measured here as importance signals
   rather than as the generation-time algorithms described.
2. Retained keys keep their original RoPE phase and generated tokens continue
   from the compressed length. Every method is treated the same way.
3. Scores are averaged over heads, which puts head-wise allocation (Ada-KV,
   DuoAttention) out of scope.

Results append to RESULTS_PATH as JSON lines and completed runs are skipped, so
a killed session resumes where it stopped.

    MODEL=Qwen/Qwen2.5-0.5B-Instruct DEVICE=cpu CTX=1024 python kv_bench.py
"""

import os, sys, json, math, time, random, itertools, gc

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config. Every value below can be set from the environment.
MODEL        = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE       = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE        = torch.float16 if DEVICE == "cuda" else torch.float32
CTX          = int(os.environ.get("CTX", 1024))      # haystack length in tokens
CHUNK        = int(os.environ.get("CHUNK", 256))     # prefill chunk for attention
OBS_WINDOW   = int(os.environ.get("OBS_WINDOW", 32)) # SnapKV observation window
RECENT       = int(os.environ.get("RECENT", 32))     # always-kept recent tokens
SINK         = int(os.environ.get("SINK", 4))        # always-kept attention sinks
MAX_NEW      = int(os.environ.get("MAX_NEW", 24))
SEED         = int(os.environ.get("SEED", 0))
OUTDIR       = os.environ.get("OUTDIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
RESULTS_PATH = os.path.join(OUTDIR, "kv_bench_results.jsonl")

METHODS = os.environ.get(
    "METHODS", "full,streamingllm,h2o,snapkv,pyramidkv,keydiff").split(",")
RATIOS  = [float(x) for x in
           os.environ.get("RATIOS", "1.0,0.5,0.3,0.2,0.1").split(",")]
DEPTHS  = [float(x) for x in
           os.environ.get("DEPTHS", "0.0,0.25,0.5,0.75,1.0").split(",")]
N_TRIALS = int(os.environ.get("N_TRIALS", 3))  # distinct needles per cell

# Decoy facts sharing the needle's surface form. At 0 the needle is the only
# distinctive span in the document, which favours a distinctiveness signal.
DISTRACTORS = int(os.environ.get("DISTRACTORS", 0))

# Structure of the filler text, as opposed to how much of it there is.
#   plain  filler sampled with replacement from ten sentences, one fact form.
#   rich   filler sampled without replacement, fact surface form drawn at random.
# The decoy count is the same in both. plain reproduces the original haystack
# exactly, including the sequence of calls made on rng.
HAYSTACK = os.environ.get("HAYSTACK", "plain")
if HAYSTACK not in ("plain", "rich"):
    raise SystemExit("HAYSTACK must be 'plain' or 'rich', not %r" % HAYSTACK)

# Experimental conditions, see make_sample().
#   qlast  question after the document, so attention-window policies see it.
#   qfirst question before the document, so the same policies cannot.
#   reuse  compress while asking about needle A, then ask about needle B.
CONDITIONS = os.environ.get("CONDITIONS", "qlast").split(",")

# Bit width used by the quantisation family, derived from the memory ratio so
# that a quantised row and an eviction row at the same `ratio` cost the same.
QUANT_MIN_BITS = int(os.environ.get("QUANT_MIN_BITS", 2))
QUANT_MAX_BITS = int(os.environ.get("QUANT_MAX_BITS", 8))

# Length of the contiguous run BUZZ keeps around each interval's local maximum.
# Must exceed a fact's token length or multi-token spans are fragmented.
BUZZ_SEG = int(os.environ.get("BUZZ_SEG", 16))

# Task set. 'synthetic' uses the generator in this file; 'ruler' and 'loogle'
# load published benchmarks.
TASKSET       = os.environ.get("TASKSET", "synthetic")
RULER_REPO    = os.environ.get("RULER_REPO", "simonjegou/ruler")
RULER_CONFIG  = os.environ.get("RULER_CONFIG", "4096")   # 4096 | 8192 | 16384
# Two RULER mirrors exist and they are not shaped the same.
#   columns  simonjegou/ruler, three separate fields, up to a 16,384 config.
#   blob     SaylorTwift/RULER-<len>-<tokenizer>, one input field, task as the
#            split, reaching 32,768 and beyond.
# The context/question boundary is the experiment, so blob rows are split back
# apart by RULER_BLOB_MARKERS below. Use the -llama-3.2-tokenizer mirrors:
# ruler_prompt() applies the chat template itself, so a -chat-template mirror
# would be wrapped twice.
RULER_SCHEMA  = os.environ.get("RULER_SCHEMA", "auto")   # auto | columns | blob

# LooGLE: real documents rather than a generated haystack. It ships context and
# question as separate fields, which is what the visibility condition needs.
# LOOGLE_CTX truncates each document to a token budget so the result is
# comparable with a RULER column; 0 leaves natural length, which is not.
LOOGLE_REPO   = os.environ.get("LOOGLE_REPO", "bigai-nlco/LooGLE")
LOOGLE_TASK   = os.environ.get("LOOGLE_TASK", "shortdep_qa")
LOOGLE_SAMPLES = int(os.environ.get("LOOGLE_SAMPLES", 50))
LOOGLE_CTX    = int(os.environ.get("LOOGLE_CTX", 16384))
LOOGLE_MAX_NEW = int(os.environ.get("LOOGLE_MAX_NEW", 64))
# One question per document. Consecutive rows repeat the same context, and the
# aggregate clusters on the document, so documents have to be what varies.
LOOGLE_ONE_PER_DOC = int(os.environ.get("LOOGLE_ONE_PER_DOC", 1))
RULER_TASKS   = [t for t in os.environ.get("RULER_TASKS", "").split(",") if t]
RULER_SAMPLES = int(os.environ.get("RULER_SAMPLES", 10))
RULER_MAX_NEW = int(os.environ.get("RULER_MAX_NEW", 128))
# Local smoke-testing only: truncate each context to N tokens so the CPU run
# does not have to hold a 4K attention matrix. Never set this on the GPU run.
RULER_TRUNC   = int(os.environ.get("RULER_TRUNC", 0))
# Share one prefill across every policy scored on the same document. Set to 0
# for the old path; test_prefill_reuse.py checks the two generate identically.
PREFILL_REUSE = int(os.environ.get("PREFILL_REUSE", 1))
# Split the sweep across workers: SHARD of NSHARDS takes every NSHARDS-th
# (condition, sample) group, each with its own OUTDIR. See run_sharded.py.
# Reduce each layer's attention to the three [S] signals inside the layer
# instead of letting the model return all of them.
ATTN_HOOKS    = int(os.environ.get("ATTN_HOOKS", 1))
NSHARDS       = max(1, int(os.environ.get("NSHARDS", 1)))
SHARD         = int(os.environ.get("SHARD", 0))
if not 0 <= SHARD < NSHARDS:
    raise SystemExit("SHARD=%d outside 0..%d" % (SHARD, NSHARDS - 1))

# Where the question sits relative to the compressed region.
#   qhidden   question forwarded onto the already compressed cache, so every
#             policy is query-agnostic. This is kvpress's default.
#   qvisible  question inside the compressed region (kvpress --query_aware).
#   qdecoy    an unrelated question of near-identical length sits where
#             qvisible puts the real one, and the real question is forwarded
#             afterwards as under qhidden. Holds the position and the size of
#             the scored prefix fixed and changes only whether the text the
#             observation window reads is the question being asked, which is
#             what separates a positional effect from a semantic one.
RULER_CONDITIONS = os.environ.get("RULER_CONDITIONS", "qhidden").split(",")
_RULER_COND_OK = ("qhidden", "qvisible", "qdecoy")
for _c in RULER_CONDITIONS:
    if _c not in _RULER_COND_OK:
        raise SystemExit("RULER_CONDITIONS: %r is not one of %s"
                         % (_c, ", ".join(_RULER_COND_OK)))

# Cache accessors. The DynamicCache API has changed across transformers
# versions, so read and write through these and nothing else.
def cache_layers(cache):
    """Return list of (key, value) tensors, each [B, H_kv, S, D]."""
    if cache is None:
        return []
    if isinstance(cache, (list, tuple)):
        return [(k, v) for k, v in cache]
    if hasattr(cache, "layers") and cache.layers is not None:
        try:
            return [(l.keys, l.values) for l in cache.layers]
        except AttributeError:
            pass
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    raise TypeError("unrecognised cache type: %r" % type(cache))


def set_layer(cache, i, k, v):
    """Write (k, v) back into layer i, in place. Covers both cache flavours."""
    if hasattr(cache, "layers") and cache.layers is not None:
        try:
            cache.layers[i].keys = k
            cache.layers[i].values = v
            return
        except AttributeError:
            pass
    if hasattr(cache, "key_cache"):
        cache.key_cache[i] = k
        cache.value_cache[i] = v
        return
    raise TypeError("cannot write to cache type %r" % type(cache))


def rebind_cache(cache, kvs):
    """
    Restore the uncompressed key/value tensors into `cache` and make its length
    bookkeeping agree with them.

    Called before every cell when one prefill is shared across policies. The
    tensors are never written in place -- index_select, clone and cat all
    allocate -- so the uncompressed view survives the previous cell and is
    restored by rebinding. The counters below are reset because one left over
    from that cell's decode would describe a cache that no longer exists;
    cache_seq_len() reads the tensors and is already safe.
    """
    for li, (k, v) in enumerate(kvs):
        set_layer(cache, li, k, v)
    S = max(int(k.shape[2]) for k, _ in kvs)
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = S
    if getattr(cache, "layers", None) is not None:
        for layer in cache.layers:
            for attr in ("_seen_tokens", "cumulative_length"):
                if hasattr(layer, attr):
                    try:
                        setattr(layer, attr, S)
                    except AttributeError:
                        pass
    return cache


def install_ragged_mask_patch(model):
    """
    PyramidKV gives each layer a different cache length. Llama builds ONE causal
    mask for every layer from the global cache length, so a ragged cache throws
    a shape error.

    In this harness batch=1 and nothing is padded, so the correct mask is simply
    all-zeros of width (this layer's cache length + q_len). We wrap each
    attention module and resize the mask when it does not match. This is a no-op
    whenever every layer has the same length.

    Returns True if the patch installed.
    """
    try:
        layers = model.model.layers
    except AttributeError:
        return False
    n = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "_ragged_patched", False):
            continue
        orig = attn.forward

        def make(attn_mod, _orig=orig):
            def fwd(*args, **kw):
                am = kw.get("attention_mask", None)
                pkv = kw.get("past_key_values", kw.get("past_key_value", None))
                hs = kw.get("hidden_states", args[0] if args else None)
                if am is not None and pkv is not None and hs is not None:
                    try:
                        li = attn_mod.layer_idx
                        past = cache_layers(pkv)[li][0].shape[2]
                        q = hs.shape[1]
                        want = past + q
                        if am.shape[-1] != want:
                            kw["attention_mask"] = torch.zeros(
                                am.shape[0], 1, q, want,
                                dtype=am.dtype, device=am.device)
                    except Exception:
                        pass
                return _orig(*args, **kw)
            return fwd

        attn.forward = make(attn)
        attn._ragged_patched = True
        n += 1
    return n > 0


def cache_seq_len(cache):
    """
    Cache length read from the tensors, so it is never stale.
    Uses the MAX over layers: with PyramidKV the layers are ragged, and the
    per-layer mask patch narrows the mask back down for the shorter ones.
    """
    return max(int(k.shape[2]) for k, _ in cache_layers(cache))


# Needle-in-a-haystack data, generated here so a run needs no download.
FILLER = [
    "The committee reviewed the quarterly figures and noted a modest improvement.",
    "Rainfall in the northern districts remained close to the seasonal average.",
    "Operators reported no interruption to service during the maintenance window.",
    "The archive was reorganised so that older records could be retrieved faster.",
    "Attendance at the regional meetings has held steady for three consecutive years.",
    "A revised timetable was circulated to all departments before the deadline.",
    "The survey instrument was piloted with a small group before wider release.",
    "Shipping volumes recovered slowly after the disruption earlier in the year.",
    "Library usage statistics show a gradual shift toward electronic resources.",
    "The laboratory replaced two ageing instruments during the summer shutdown.",
]
CITIES = ["Mumbai", "Lisbon", "Nagoya", "Calgary", "Bergen", "Tampere", "Pune", "Utrecht"]


def _fact(city, number):
    return "The special magic number for %s is %s." % (city, number)


def _question(city):
    return ("What is the special magic number for %s? "
            "Answer with the number only." % city)


def _wrap(tokenizer, document, question, question_first):
    """Assemble the chat prompt with the question before or after the body."""
    if question_first:
        content = ("Answer the question using the document that follows.\n\n"
                   "%s\n\n<document>\n%s\n</document>" % (question, document))
    else:
        content = ("Read the following document and answer the question.\n\n"
                   "<document>\n%s\n</document>\n\n%s" % (document, question))
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return content + "\nAnswer: "


# Rich-haystack vocabulary. Four slots, so the cross product is 230,400
# distinct sentences and a document of a few hundred needs none of them twice.
_RF_WHO = [
    "The regional office", "The audit team", "A visiting inspector",
    "The night shift", "The procurement desk", "Two of the engineers",
    "The duty supervisor", "The records clerk", "A contracted surveyor",
    "The planning group", "The safety officer", "The warehouse crew",
    "An external assessor", "The dispatch desk", "The finance team",
    "The maintenance lead", "A junior technician", "The standards board",
    "The receiving bay", "The training coordinator", "A relief operator",
    "The quality panel", "The scheduling office", "The site manager",
]
_RF_DID = [
    "revised", "archived", "circulated", "re-indexed", "signed off",
    "queried", "consolidated", "withdrew", "annotated", "reissued",
    "cross-checked", "shortened", "deferred", "duplicated", "approved",
    "rejected", "reconciled", "transcribed", "flagged", "retired",
]
_RF_WHAT = [
    "the maintenance schedule", "the shipping manifest", "the duty roster",
    "the inspection log", "the calibration sheet", "the delivery note",
    "the access register", "the incident summary", "the parts list",
    "the handover notes", "the service contract", "the floor plan",
    "the training record", "the stock reconciliation", "the visitor book",
    "the fuel return", "the equipment inventory", "the shift report",
    "the purchase requisition", "the disposal certificate",
    "the boundary survey", "the insurance schedule", "the pressure readings",
    "the drainage report",
]
_RF_WHEN = [
    "before the quarter closed", "after the summer shutdown",
    "during the annual audit", "at the start of the second week",
    "ahead of the site visit", "once the backlog cleared",
    "on the morning of the handover", "while the line was stopped",
    "in the week the contract began", "after the second reminder",
    "before the figures were published", "on the day the permit lapsed",
    "between the two inspections", "as the shift changed",
    "after the roof was repaired", "when the new system went live",
    "before the deadline was extended", "during the cold spell",
    "once the invoices were matched", "at the end of the trial period",
]

# Every form names the city and the phrase the question asks about, so no form
# is easier to spot than another by shape alone.
_FACT_FORMS = [
    "The special magic number for %s is %s.",
    "For %s, the special magic number is %s.",
    "%s has been assigned the special magic number %s.",
    "The records list the special magic number for %s as %s.",
    "In the case of %s, the special magic number recorded is %s.",
]


def _rich_filler(rng, n):
    """`n` distinct filler sentences. Distinct is the point: the plain filler
    repeats ten sentences, which leaves every fact-bearing span the only
    unusual text in the document and is the best possible case for a signal
    that scores distinctiveness."""
    seen, out = set(), []
    while len(out) < n:
        k = (rng.randrange(len(_RF_WHO)), rng.randrange(len(_RF_DID)),
             rng.randrange(len(_RF_WHAT)), rng.randrange(len(_RF_WHEN)))
        if k in seen:
            continue
        seen.add(k)
        out.append("%s %s %s %s." % (_RF_WHO[k[0]], _RF_DID[k[1]],
                                     _RF_WHAT[k[2]], _RF_WHEN[k[3]]))
    return out


def _fact_rich(rng, city, number):
    return rng.choice(_FACT_FORMS) % (city, number)


def _make_sample_rich(tokenizer, ctx_tokens, depth, rng, condition="qlast"):
    """The rich-haystack counterpart of make_sample(). Same contract, same
    conditions, same decoy count; only the filler and the surface forms
    differ."""
    cities = list(CITIES)
    rng.shuffle(cities)
    city, number = cities[0], str(rng.randint(100000, 999999))

    probe = _rich_filler(rng, 1)[0]
    per = max(1, len(tokenizer(probe + " ", add_special_tokens=False)["input_ids"]))
    n_sent = max(4, ctx_tokens // per)
    body = _rich_filler(rng, n_sent)

    for i in range(min(DISTRACTORS, len(cities) - 2)):
        d_city = cities[i + 2]
        d_num = str(rng.randint(100000, 999999))
        body.insert(rng.randrange(len(body) + 1), _fact_rich(rng, d_city, d_num))

    def place(text, frac):
        pos = min(len(body), max(0, int(round(frac * len(body)))))
        body.insert(pos, text)

    followup = None
    if condition == "reuse":
        a_city, a_num = cities[1], str(rng.randint(100000, 999999))
        place(_fact_rich(rng, a_city, a_num), 0.5)
        place(_fact_rich(rng, city, number), depth)
        question = _question(a_city)
    else:
        place(_fact_rich(rng, city, number), depth)
        question = _question(city)

    document = " ".join(body)
    prompt = _wrap(tokenizer, document, question,
                   question_first=(condition == "qfirst"))
    if condition == "reuse":
        followup = _followup_suffix(tokenizer, document, question,
                                    _question(city))
    return prompt, number, followup


def make_sample(tokenizer, ctx_tokens, depth, rng, condition="qlast"):
    """
    Return (prompt, answer, followup) where `followup` is None except in the
    reuse condition, in which case it is the text of a SECOND question to be
    asked of the already-compressed cache.

    In the reuse condition the needle the compression question targets is
    pinned at mid-document and the needle actually asked about is the one that
    moves with `depth`, so the depth axis keeps meaning the position of the
    span whose retrieval is being measured.
    """
    if HAYSTACK == "rich":
        return _make_sample_rich(tokenizer, ctx_tokens, depth, rng, condition)

    cities = list(CITIES)
    rng.shuffle(cities)
    city, number = cities[0], str(rng.randint(100000, 999999))

    sent = rng.choice(FILLER)
    per = max(1, len(tokenizer(sent + " ", add_special_tokens=False)["input_ids"]))
    n_sent = max(4, ctx_tokens // per)
    body = [rng.choice(FILLER) for _ in range(n_sent)]

    # Decoy facts: same surface form, different city and number, scattered
    # uniformly. The question names only the true city.
    for i in range(min(DISTRACTORS, len(cities) - 2)):
        d_city = cities[i + 2]
        d_num = str(rng.randint(100000, 999999))
        body.insert(rng.randrange(len(body) + 1), _fact(d_city, d_num))

    def place(text, frac):
        pos = min(len(body), max(0, int(round(frac * len(body)))))
        body.insert(pos, text)

    followup = None
    if condition == "reuse":
        # needle A drives compression, needle B is what we actually ask for
        a_city, a_num = cities[1], str(rng.randint(100000, 999999))
        place(_fact(a_city, a_num), 0.5)
        place(_fact(city, number), depth)
        question = _question(a_city)          # compression sees THIS question
    else:
        place(_fact(city, number), depth)
        question = _question(city)

    document = " ".join(body)
    prompt = _wrap(tokenizer, document, question,
                   question_first=(condition == "qfirst"))

    if condition == "reuse":
        # The follow-up has to be a new user turn. The prompt already ends with the
        # assistant-turn opener, so a bare question appended here lands inside that
        # turn and the model emits end-of-turn at once, scoring 0% for every method.
        followup = _followup_suffix(tokenizer, document, question,
                                    _question(city))
    return prompt, number, followup


def _followup_suffix(tokenizer, document, question, followup):
    """
    The exact text to append to a prompt so that `followup` becomes a second
    user turn. Computed as the delta between the one-turn and three-turn
    renderings of the same conversation, so no chat format is hardcoded.
    """
    base = ("Read the following document and answer the question.\n\n"
            "<document>\n%s\n</document>\n\n%s" % (document, question))
    try:
        one = tokenizer.apply_chat_template(
            [{"role": "user", "content": base}],
            tokenize=False, add_generation_prompt=True)
        three = tokenizer.apply_chat_template(
            [{"role": "user", "content": base},
             {"role": "assistant", "content": ""},
             {"role": "user", "content": followup}],
            tokenize=False, add_generation_prompt=True)
        if three.startswith(one):
            return three[len(one):]
    except Exception:
        pass
    # Fallback for a tokenizer without a usable chat template.
    return "\n\n" + followup + "\nAnswer: "


# RULER (Hsieh et al. 2024), from the pre-generated copy kvpress evaluates
# against. Fields: context, question, answer_prefix, answer, task, max_new_tokens.
RULER_ALL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]

# Sentinels cut the rendered chat template at an exact point. Searching for the
# context itself is fragile: the question can repeat a phrase from it.
_SENT_C = "\u0001RULERCTX\u0001"
_SENT_Q = "\u0001RULERQST\u0001"


# Derived by derive_ruler_templates.py from config 4096. The templates do not
# vary with length, so markers derived at 4K place the same boundary at 32K.
RULER_BLOB_MARKERS = {
    'niah_single_1': ('What is the special magic number for ',
                      'The special magic number for '),
    'niah_single_2': ('What is the special magic number for ',
                      'The special magic number for '),
    'niah_single_3': ('What is the special magic uuid for ',
                      'The special magic uuid for '),
    'niah_multikey_1': ('What is the special magic number for ',
                        'The special magic number for '),
    'niah_multikey_2': ('What is the special magic number for ',
                        'The special magic number for '),
    'niah_multikey_3': ('What is the special magic uuid for ',
                        'The special magic uuid for '),
    'niah_multivalue': ('What are all the special magic numbers for ',
                        'The special magic numbers for '),
    'niah_multiquery': ('What are all the special magic numbers for ',
                        'The special magic numbers for '),
    # Stops before the 'assgined'/'assigned' word: the three-field mirror still
    # carries the upstream typo and the blob mirror does not.
    'vt': ('Question: Find all variables that are assigned the value ',
           'Answer: According to the chain(s) of variable assignment in the text '
           'above, 5 variables are '),
    'cwe': ('Question: What are the 10 most common words in the above list? ',
            'Answer: The top 10 words that appear most often in the list are:'),
    'fwe': ("Question: Do not provide any explanation. Please ignore the dots "
            "'....'. What are the three most frequently appeared words in the "
            "above coded text? ",
            'Answer: According to the coded text above, the three most frequently '
            'appeared words are:'),
    'qa_1': ('Answer the question based on the given documents. Only give me the '
             'answer and do not output any other words.\n\nQuestion: ',
             'Answer:'),
    'qa_2': ('Answer the question based on the given documents. Only give me the '
             'answer and do not output any other words.\n\nQuestion: ',
             'Answer:'),
}

# RULER's own per-task generation budget, which the blob mirrors do not carry.
# RULER_MAX_NEW caps these, it does not replace them.
RULER_BLOB_MAXNEW = {
    'niah_single_1': 128, 'niah_single_2': 128, 'niah_single_3': 128,
    'niah_multikey_1': 128, 'niah_multikey_2': 128, 'niah_multikey_3': 128,
    'niah_multivalue': 128, 'niah_multiquery': 128,
    'vt': 30, 'cwe': 120, 'fwe': 50, 'qa_1': 32, 'qa_2': 32,
}


def ruler_blob_split(task, text):
    """Recover (context, question, answer_prefix) from one concatenated `input`.

    Both boundaries are placed at the LAST occurrence of the task's marker, not
    the first, so a marker that also appears somewhere inside the haystack
    cannot pull the boundary forward. A missing marker raises rather than
    guessing: a silently misplaced boundary would move the question across the
    compression boundary and quietly measure the wrong condition, which is the
    one failure this whole path exists to avoid.
    """
    try:
        q_mark, a_mark = RULER_BLOB_MARKERS[task]
    except KeyError:
        raise ValueError("no blob markers for RULER task %r" % task)
    qi = text.rfind(q_mark)
    if qi < 0:
        raise ValueError("RULER %s: question marker %r not found in a %d-character "
                         "row; the mirror's template may differ from the one the "
                         "markers were derived from" % (task, q_mark[:50], len(text)))
    context, tail = text[:qi], text[qi:]
    ai = tail.rfind(a_mark)
    if ai <= 0:
        raise ValueError("RULER %s: answer-prefix marker %r not found after the "
                         "question" % (task, a_mark[:50]))
    return context, tail[:ai], tail[ai:]


def _ruler_rows_columns(want):
    """{task: [row, ...]} from a mirror carrying the three fields separately."""
    from datasets import load_dataset
    ds = load_dataset(RULER_REPO, RULER_CONFIG, split="test")
    by_task = {t: [] for t in want}
    for row in ds:
        t = row["task"]
        if t in by_task and len(by_task[t]) < RULER_SAMPLES:
            by_task[t].append(dict(context=row["context"],
                                   question=row["question"],
                                   answer_prefix=row["answer_prefix"],
                                   answers=list(row["answer"]),
                                   max_new_tokens=int(row["max_new_tokens"])))
        if all(len(v) >= RULER_SAMPLES for v in by_task.values()):
            break
    return by_task


def _ruler_rows_blob(want):
    """Same, from a mirror whose splits are tasks and whose rows are one blob.

    Streamed. A 32,768-token split is ~60 MB and only RULER_SAMPLES rows of it
    are wanted, so downloading all thirteen splits in full would spend minutes
    of a GPU session on bytes that are then discarded.
    """
    from datasets import load_dataset
    by_task = {t: [] for t in want}
    for t in want:
        try:
            ds = load_dataset(RULER_REPO, split=t, streaming=True)
        except Exception as e:
            print("  WARNING: RULER split %r not streamable (%s); falling back"
                  % (t, type(e).__name__), flush=True)
            ds = load_dataset(RULER_REPO, split=t)
        for row in ds:
            ctx, q, ap = ruler_blob_split(t, row["input"])
            by_task[t].append(dict(context=ctx, question=q, answer_prefix=ap,
                                   answers=list(row["outputs"]),
                                   max_new_tokens=RULER_BLOB_MAXNEW.get(t, RULER_MAX_NEW)))
            if len(by_task[t]) >= RULER_SAMPLES:
                break
    return by_task


def ruler_load(tokenizer):
    """
    Returns an ordered list of sample dicts. Samples are taken as the FIRST
    RULER_SAMPLES rows of each task, not sampled at random, so the selection is
    reproducible from the config alone and a resumed run sees the same items.
    """
    want = RULER_TASKS or RULER_ALL_TASKS
    schema = RULER_SCHEMA
    if schema == "auto":
        # Try the three-field layout first. The attempt is cheap when it fails: an
        # undefined configuration raises before anything is downloaded.
        try:
            by_task = _ruler_rows_columns(want)
            schema = "columns"
        except Exception as e:
            print("  RULER: %s has no %r column layout (%s); reading the blob "
                  "layout instead" % (RULER_REPO, RULER_CONFIG, type(e).__name__),
                  flush=True)
            by_task, schema = _ruler_rows_blob(want), "blob"
    elif schema == "columns":
        by_task = _ruler_rows_columns(want)
    elif schema == "blob":
        by_task = _ruler_rows_blob(want)
    else:
        raise ValueError("RULER_SCHEMA must be auto, columns or blob, not %r" % schema)
    print("  RULER: %s, schema %s, %d task(s)" % (RULER_REPO, schema, len(want)),
          flush=True)

    out = []
    for t in want:
        rows = by_task.get(t, [])
        if not rows:
            print("  WARNING: RULER task %r returned no rows" % t, flush=True)
        for i, row in enumerate(rows):
            ctx = row["context"]
            if RULER_TRUNC:
                ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
                ctx = tokenizer.decode(ids[:RULER_TRUNC])
            out.append(dict(task=t, index=i, context=ctx,
                            question=row["question"],
                            answer_prefix=row["answer_prefix"],
                            answers=list(row["answers"]),
                            max_new=min(int(row["max_new_tokens"]),
                                        RULER_MAX_NEW)))
    return out


def loogle_load(tokenizer):
    """LooGLE rows in the same shape ruler_load() returns.

    The question carries no answer prefix of its own, so one is supplied: the
    prompt is `context` + "Question: ..." + "Answer:", which is the same
    three-part shape RULER has and is what lets ruler_prompt() place the
    question on either side of the compression boundary without a second code
    path. The prefix is fixed, because a different prefix is a different
    measurement.
    """
    from datasets import load_dataset
    ds = load_dataset(LOOGLE_REPO, LOOGLE_TASK, split="test")
    out, seen = [], set()
    for row in ds:
        doc = str(row.get("title") or row.get("doc_id") or "")
        if LOOGLE_ONE_PER_DOC and doc in seen:
            continue
        seen.add(doc)
        ctx = row["context"]
        if LOOGLE_CTX:
            ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
            ctx = tokenizer.decode(ids[:LOOGLE_CTX])
        ans = row["answer"]
        out.append(dict(task=LOOGLE_TASK, index=len(out), context=ctx,
                        question="Question: " + str(row["question"]).strip() + "\n",
                        answer_prefix="Answer:",
                        answers=[ans] if isinstance(ans, str) else list(ans),
                        max_new=min(LOOGLE_MAX_NEW, RULER_MAX_NEW),
                        doc=doc))
        if len(out) >= LOOGLE_SAMPLES:
            break
    if len(out) < LOOGLE_SAMPLES:
        print("  WARNING: LooGLE returned %d samples, %d were asked for"
              % (len(out), LOOGLE_SAMPLES), flush=True)
    return out


_ARTICLES = ("a", "an", "the")


def _norm_answer(s):
    """SQuAD normalisation: lowercase, drop punctuation and articles, collapse
    whitespace. LooGLE's own scorer uses this before computing F1, and the
    numbers are not comparable to published ones without it."""
    import re as _re, string as _string
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(_string.punctuation))
    s = " ".join(w for w in s.split() if w not in _ARTICLES)
    return s


def loogle_score(task, prediction, answers):
    """Token-level F1 against the best reference.

    Not substring match. RULER's references are needles -- a magic number
    either appears in the generation or it does not -- while a LooGLE answer is
    a sentence a person wrote, so a model can be right in different words.
    Substring match would score a correct paraphrase zero and would make the
    naturalistic column incomparable with the published LooGLE numbers as well.
    F1 over normalised tokens is what LooGLE itself reports.
    """
    best = 0.0
    pred = _norm_answer(prediction).split()
    for a in answers:
        gold = _norm_answer(a).split()
        if not pred or not gold:
            best = max(best, float(pred == gold))
            continue
        common = {}
        for w in pred:
            common[w] = common.get(w, 0) + 1
        overlap = 0
        for w in gold:
            if common.get(w, 0) > 0:
                common[w] -= 1
                overlap += 1
        if overlap == 0:
            continue
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def ruler_prompt(tokenizer, sample, condition):
    """
    Returns (prefix_text, suffix_text). `prefix_text` is prefilled and then
    compressed; `suffix_text` is forwarded onto the compressed cache. The split
    point is what the condition controls, and it is placed by rendering the
    template around sentinels rather than by searching the rendered output.
    """
    try:
        tmpl = tokenizer.apply_chat_template(
            [{"role": "user", "content": _SENT_C + "\n\n" + _SENT_Q}],
            tokenize=False, add_generation_prompt=True)
        pre, rest = tmpl.split(_SENT_C, 1)
        mid, post = rest.split(_SENT_Q, 1)
    except Exception:
        pre, mid, post = "", "\n\n", "\n"
    post = post + sample["answer_prefix"]
    if condition == "qvisible":
        return (pre + sample["context"] + mid + sample["question"], post)
    if condition == "qdecoy":
        decoy = sample.get("decoy")
        if not decoy:
            raise RuntimeError("qdecoy needs sample['decoy']; attach_decoys() "
                               "did not run for task %r" % sample["task"])
        return (pre + sample["context"] + mid + decoy,
                mid + sample["question"] + post)
    return (pre + sample["context"], mid + sample["question"] + post)


def attach_decoys(tokenizer, samples):
    """Give each sample an unrelated question to stand in for its own.

    The decoy comes from another sample of the same task, chosen by closest
    token length, so it is the same kind of text at the same position and
    almost the same size. Length matters because the budget is a fraction of
    the prefix: a decoy much shorter or longer would change how much cache the
    policy keeps, which is the thing being held fixed.

    Deterministic, with no generator of its own; ties go to the lower index. A
    task with one sample has nothing to borrow from, and ruler_prompt() turns
    that into an error rather than a silent fallback to the real question.
    """
    by_task = {}
    for smp in samples:
        by_task.setdefault(smp["task"], []).append(smp)
    for task, group in by_task.items():
        if len(group) < 2:
            print("  WARNING: task %r has one sample, no decoy available"
                  % task, flush=True)
            continue
        lens = [len(tokenizer(s["question"], add_special_tokens=False)["input_ids"])
                for s in group]
        for i, smp in enumerate(group):
            j = min((k for k in range(len(group)) if k != i),
                    key=lambda k: (abs(lens[k] - lens[i]), k))
            smp["decoy"] = group[j]["question"]
            smp["decoy_from"] = group[j]["index"]
            smp["decoy_tokens"] = lens[j]


_CTRL = dict.fromkeys(range(0x20))


def ruler_score(task, prediction, answers):
    """
    RULER's own metric, matching kvpress's calculate_metrics: case-insensitive
    substring match of each reference against the cleaned prediction. QA tasks
    score 1 if ANY reference is found; every other task scores the FRACTION of
    references found, which is what makes the multi-value and multi-query tasks
    partial-credit rather than all-or-nothing.
    """
    pred = str(prediction).translate(_CTRL).strip().lower()
    if not answers:
        return 0.0
    found = [str(a).lower() in pred for a in answers]
    if task.split("_")[0] == "qa":
        return float(any(found))
    return float(sum(found)) / len(found)


# Prefill with chunked attention accumulation.
def _attention_modules(model):
    """The per-layer attention submodules, in layer order.

    Found by walking the decoder layers rather than by class name, so the order
    is the model's own and matches the order `out.attentions` would have used.
    Returns [] when the structure is not recognised, which makes the caller
    fall back to reading attentions off the model output.
    """
    for path in ("model.layers", "model.model.layers", "transformer.h",
                 "model.decoder.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None:
            continue
        mods = []
        for layer in obj:
            att = next((getattr(layer, n) for n in
                        ("self_attn", "attn", "attention", "self_attention")
                        if getattr(layer, n, None) is not None), None)
            if att is None:
                return []
            mods.append(att)
        if mods:
            return mods
    return []


def _module_device(module, default):
    """Device a module's parameters live on, for placing its accumulator."""
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return default


@torch.no_grad()
def prefill_with_scores(model, input_ids):
    """
    Returns (cache, sig, last_logits) where sig is a dict of per-layer signals:
      sig["acc"][layer]  : [S] attention mass each key received, summed over all
                           queries and averaged over heads  (H2O signal)
      sig["win"][layer]  : [S] attention mass from the last OBS_WINDOW queries
                           (SnapKV / PyramidKV signal)
      sig["lastq"][layer]: [S] attention from the FINAL query only (TOVA signal)
      last_logits        : [B, 1, V] logits at the final prompt position, reused
                           so we never forward the last token twice
    """
    S = input_ids.shape[1]
    cache = None
    state = {"acc": None, "win": None, "lastq": None}
    last_logits = None
    attn_mods = _attention_modules(model) if ATTN_HOOKS else []

    def reduce_layer(li, a, start, end):
        """Fold one layer's [B, H, q, k] attention into the three [S] signals.

        Runs on a's own device and is the only thing that ever reads the
        attention matrix, so the matrix never has to outlive the layer that
        produced it. Accumulators are allocated by the caller before the first
        call, because only the caller knows which devices to put them on.
        """
        a = a[0].float()                                 # [H, q, k]
        k = a.shape[-1]
        state["acc"][li][:k] += a.sum(dim=1).mean(dim=0)
        if end >= S - OBS_WINDOW:                        # final chunk(s)
            qs = max(0, (S - OBS_WINDOW) - start)
            if a.shape[1] > qs:
                state["win"][li][:k] += a[:, qs:, :].sum(dim=1).mean(dim=0)
        if end == S:                                     # the very last query row
            state["lastq"][li][:k] = a[:, -1, :].mean(dim=0)

    def alloc(devices):
        # One accumulator per layer, on that layer's device. Under a device_map,
        # layer i's attention and its keys live on whichever card holds layer i.
        for key in ("acc", "win", "lastq"):
            state[key] = [torch.zeros(S, dtype=torch.float32, device=d)
                          for d in devices]

    for start in range(0, S, CHUNK):
        end = min(S, start + CHUNK)
        handles, span = [], (start, end)
        if attn_mods:
            # Reduce each layer's attention inside the layer and hand back None in its
            # place. Otherwise accelerate gathers all 32 tensors onto the input device,
            # which is 8 GiB at 16,384 tokens.
            if state["acc"] is None:
                alloc([_module_device(m, input_ids.device) for m in attn_mods])

            def mk(li):
                def hook(module, args, output):
                    if not (isinstance(output, tuple) and len(output) > 1):
                        return output
                    a = output[1]
                    if a is None:
                        return output
                    reduce_layer(li, a, span[0], span[1])
                    return (output[0], None) + tuple(output[2:])
                return hook

            handles = [m.register_forward_hook(mk(i))
                       for i, m in enumerate(attn_mods)]
        try:
            # output_attentions stays off while the hooks do the work. The attention module
            # returns its weights as output[1] either way, so the hook still sees them and
            # transformers collects nothing.
            out = model(input_ids=input_ids[:, start:end],
                        past_key_values=cache,
                        use_cache=True,
                        output_attentions=not attn_mods,
                        attention_mask=torch.ones(1, end, dtype=torch.long,
                                                  device=input_ids.device),
                        cache_position=torch.arange(start, end,
                                                    device=input_ids.device))
        finally:
            for h in handles:
                h.remove()
        cache = out.past_key_values
        last_logits = out.logits[:, -1:].clone()

        if not attn_mods:
            attns = out.attentions                  # per layer [B, H, q, k]
            if state["acc"] is None:
                alloc([a.device for a in attns])
            for li, a in enumerate(attns):
                reduce_layer(li, a, start, end)
            del attns
        del out
        gc.collect()
    return (cache, {"acc": state["acc"], "win": state["win"],
                    "lastq": state["lastq"]}, last_logits)


# Selection policies. Each returns the indices to keep, sorted ascending.
def _protect(S):
    """Indices that every policy keeps: attention sinks + recent window."""
    keep = set(range(min(SINK, S)))
    keep |= set(range(max(0, S - RECENT), S))
    return keep


def _topk_with_protected(score, budget, S):
    """Keep the protected indices plus the highest-scoring of the rest."""
    score = score[:S].clone().float()
    prot = torch.zeros(S, dtype=torch.bool, device=score.device)
    prot[:min(SINK, S)] = True
    prot[max(0, S - RECENT):] = True
    n_prot = int(prot.sum())
    budget = max(n_prot, min(int(budget), S))
    free = budget - n_prot
    keep = prot.clone()
    if free > 0:
        score = score.masked_fill(prot, float("-inf"))
        idx = torch.topk(score, k=min(free, S - n_prot)).indices
        keep[idx] = True
    return torch.nonzero(keep, as_tuple=True)[0]


def select_indices(method, layer, n_layers, S, budget, sig, keys, vals):
    dev = keys[layer].device
    acc, win, lastq = sig["acc"], sig["win"], sig["lastq"]

    if method == "full":
        return torch.arange(S, device=dev)

    if method == "streamingllm":
        # attention sinks plus one contiguous recent window filling the budget
        b = max(SINK + 1, min(int(budget), S))
        n_sink = min(SINK, S)
        keep = torch.zeros(S, dtype=torch.bool, device=dev)
        keep[:n_sink] = True
        keep[max(0, S - (b - n_sink)):] = True
        return torch.nonzero(keep, as_tuple=True)[0]

    if method == "h2o":
        return _topk_with_protected(acc[layer], budget, S)

    if method == "snapkv":
        s = win[layer].clone()
        s = F.avg_pool1d(s[None, None, :], kernel_size=7, stride=1, padding=3)[0, 0]
        return _topk_with_protected(s, budget, S)

    if method == "pyramidkv":
        # more budget to lower layers, less to higher; same total as uniform
        lo, hi = 1.7, 0.3
        frac = lo + (hi - lo) * (layer / max(1, n_layers - 1))
        b = int(round(budget * frac))
        s = win[layer].clone()
        s = F.avg_pool1d(s[None, None, :], kernel_size=7, stride=1, padding=3)[0, 0]
        return _topk_with_protected(s, b, S)

    if method == "keydiff":
        # geometrically distinctive keys: lowest cosine similarity to mean key.
        # Uses no attention at all, which is KeyDiff's stated advantage.
        k = keys[layer][0].float()                   # [H_kv, S, D]
        k = k.mean(dim=0)                            # [S, D]
        kn = F.normalize(k, dim=-1)
        mu = F.normalize(kn.mean(dim=0, keepdim=True), dim=-1)
        sim = (kn @ mu.T).squeeze(-1)                # [S]
        return _topk_with_protected(-sim, budget, S)

    if method == "tova":
        # Oren et al.: drop the state receiving least attention from the most recent
        # query. One query row rather than a window, which separates it from SnapKV.
        return _topk_with_protected(lastq[layer], budget, S)

    if method == "valueaware":
        # Guo et al.: weight attention by value norm, since attention alone misranks
        # entries with small attention and a large value vector.
        v = vals[layer][0].float()                   # [H_kv, S, D]
        vnorm = v.norm(dim=-1).mean(dim=0)           # [S]
        # acc and vals can sit on different cards under device_map. This is the only
        # method multiplying a score by a cache-derived quantity, so the only one
        # that can hit the mismatch.
        return _topk_with_protected(
            acc[layer][:S].to(dev) * vnorm[:S].to(dev), budget, S)

    if method == "buzz":
        # Zhao et al.: take the local attention maximum within each interval rather
        # than a global top-k, so the whole history is represented.
        # Retention has to be segmented. A fact spans several tokens, and keeping
        # only the highest-scoring one fragments it so it cannot be read back.
        prot = torch.zeros(S, dtype=torch.bool, device=dev)
        prot[:min(SINK, S)] = True
        prot[max(0, S - RECENT):] = True
        n_prot = int(prot.sum())
        budget = max(n_prot, min(int(budget), S))
        free = budget - n_prot
        lo, hi = min(SINK, S), max(0, S - RECENT)
        keep = prot.clone()
        if free > 0 and hi > lo:
            score = acc[layer][:S].clone().float()
            seg = max(1, int(BUZZ_SEG))
            n_seg = max(1, free // seg)
            edges = torch.linspace(lo, hi, steps=n_seg + 1).long().tolist()
            half = seg // 2
            for a, b in zip(edges[:-1], edges[1:]):
                if b <= a:
                    continue
                c = a + int(torch.argmax(score[a:b]))
                keep[max(lo, c - half):min(hi, c - half + seg)] = True
        return torch.nonzero(keep, as_tuple=True)[0]

    raise ValueError("unknown method: %s" % method)


# Numerical compression: nothing is discarded, entries are stored in fewer bits.
# Fake quantisation (quantise then dequantise in place) reproduces the accuracy
# effect without a custom kernel. An entry at b bits costs b/16 of an fp16 one,
# so a quantised row and an eviction row at the same ratio match on memory.
def _fake_quant(x, bits, dim):
    """Uniform min/max affine quantisation along `dim`, returned dequantised."""
    qmax = float(2 ** bits - 1)
    mn = x.amin(dim=dim, keepdim=True)
    mx = x.amax(dim=dim, keepdim=True)
    scale = (mx - mn).clamp(min=1e-8) / qmax
    q = torch.round((x - mn) / scale).clamp_(0, qmax)
    return q * scale + mn


def bits_for_ratio(ratio):
    """fp16 baseline, so `ratio` of the memory is 16*ratio bits per entry."""
    return max(QUANT_MIN_BITS, min(QUANT_MAX_BITS, int(round(16.0 * ratio))))


@torch.no_grad()
def quantise_cache(cache, ratio):
    """
    KIVI: keys quantised PER CHANNEL, values PER TOKEN, on the asymmetry that
    the key cache has structured outliers along particular channels while the
    value cache does not. The most recent RECENT tokens are left in full
    precision, which is KIVI's residual window.
    """
    bits = bits_for_ratio(ratio)
    for li, (k, v) in enumerate(cache_layers(cache)):
        S = k.shape[2]
        split = max(0, S - RECENT)
        if split == 0:
            continue
        kq, vq = k.clone(), v.clone()
        # keys: group along the sequence for each channel  -> dim=2
        kq[:, :, :split, :] = _fake_quant(k[:, :, :split, :].float(), bits, dim=2).to(k.dtype)
        # values: group along the channel for each token   -> dim=-1
        vq[:, :, :split, :] = _fake_quant(v[:, :, :split, :].float(), bits, dim=-1).to(v.dtype)
        set_layer(cache, li, kq, vq)
    return cache, bits


QUANT_METHODS = {"kivi"}


@torch.no_grad()
def compress(cache, method, S, ratio, sig):
    """
    Returns (cache, effective_entries). For eviction policies that is the mean
    number of entries retained. For the quantisation family nothing is
    discarded, so the comparable quantity is the memory-equivalent entry count:
    S entries at b bits cost the same as S*b/16 entries at fp16.
    """
    if method == "full" or ratio >= 1.0:
        return cache, S
    if method in QUANT_METHODS:
        cache, bits = quantise_cache(cache, ratio)
        return cache, S * bits / 16.0
    kvs = cache_layers(cache)
    n_layers = len(kvs)
    budget = max(SINK + RECENT + 1, int(round(ratio * S)))
    keys = [k for k, _ in kvs]
    vals = [v for _, v in kvs]
    total = 0
    for li, (k, v) in enumerate(kvs):
        idx = select_indices(method, li, n_layers, S, budget, sig, keys, vals)
        idx = idx.to(k.device, dtype=torch.long)
        set_layer(cache, li, k.index_select(2, idx), v.index_select(2, idx))
        total += int(idx.numel())
    return cache, total / n_layers


# Decode: a manual greedy loop, which keeps generate()'s cache bookkeeping out.
_NONFINITE_SEEN = [0]


def check_finite(logits, where):
    """Fail loudly on non-finite logits instead of silently decoding token 0.

    argmax over a vector containing NaN or inf returns index 0 on CUDA, and
    index 0 is a printable character in every vocabulary this harness uses
    ("!" in Qwen's). The run therefore does not crash: it produces fluent-
    looking text with characters injected at random positions, scores badly,
    and reports completion. That is exactly what happened to the Qwen2.5-7B
    sweep of 2026-11-13, which burned 7.7 hours of quota and corrupted 59 per
    cent of its rows -- including 48 per cent of the UNCOMPRESSED rows -- before
    anyone looked at the strings. The cause is running a bfloat16-trained model
    in float16 on a T4, which has no hardware bfloat16; the cost was entirely in
    not noticing.

    Raising here turns a silent eight-hour waste into a first-minute failure.
    It is deliberately an exception and not a warning: a results file that
    reaches analysis with corrupted rows in it is worse than no file, because
    the corruption looks like a finding about model scale.
    """
    if torch.isfinite(logits).all():
        return
    _NONFINITE_SEEN[0] += 1
    raise RuntimeError(
        "non-finite logits at %s: the model is numerically unstable in %s. "
        "argmax would return token id 0 and the run would complete with "
        "corrupted text. If this model was trained in bfloat16 it cannot be "
        "run in float16 on this hardware; pick a float16-safe model or a "
        "device with bfloat16 support." % (where, DTYPE))


@torch.no_grad()
def decode(model, tokenizer, cache, first_logits, orig_len, max_new=MAX_NEW):
    """
    Greedy decode from a (possibly compressed) cache.

    Two different position quantities have to be kept apart, and conflating
    them is what makes a compressed cache produce garbage:

      cache_position -- where the new entry is written, and how wide the causal
                        mask is. Must follow the PHYSICAL (compressed) length.
      position_ids   -- the RoPE phase of the new token. Must continue from the
                        ORIGINAL sequence length, because eviction does not
                        re-index the retained keys: they keep the phase they
                        were given at prefill. Using the compressed length here
                        puts the query ~450 positions away from keys it should
                        be adjacent to, and retrieval collapses to zero.
    """
    toks = []
    check_finite(first_logits[:, -1], "prefill")
    nxt = first_logits[:, -1].argmax(-1, keepdim=True)
    dev = nxt.device
    eos = tokenizer.eos_token_id
    for step in range(max_new):
        tid = int(nxt[0, 0])
        if eos is not None and tid == eos:
            break
        toks.append(tid)
        phys = cache_seq_len(cache)
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                    attention_mask=torch.ones(1, phys + 1, dtype=torch.long,
                                              device=dev),
                    cache_position=torch.tensor([phys], device=dev),
                    position_ids=torch.tensor([[orig_len + step]], device=dev))
        cache = out.past_key_values
        check_finite(out.logits[:, -1], "decode step %d" % step)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    # Pinned rather than left to the tokenizer default, which is not the same across
    # the models compared here: Qwen2.5 ships False, Llama-3.1 ships True.
    return tokenizer.decode(toks, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


@torch.no_grad()
def extend_with_text(model, tokenizer, cache, text, orig_len):
    """
    Forward `text` onto an already-compressed cache and return
    (cache, logits, new_orig_len).

    This is what makes the reuse condition a real test rather than a re-run:
    the cache was compressed while a DIFFERENT question was visible, and is now
    asked to serve this one without being rebuilt. Positions follow the same
    rule as decode() -- cache_position tracks the physical compressed length,
    position_ids continue from the original uncompressed length, because
    eviction never re-indexed the retained keys.
    """
    ids = tokenizer(text, add_special_tokens=False,
                    return_tensors="pt").input_ids.to(next(model.parameters()).device)
    L = ids.shape[1]
    phys = cache_seq_len(cache)
    dev = ids.device
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                attention_mask=torch.ones(1, phys + L, dtype=torch.long, device=dev),
                cache_position=torch.arange(phys, phys + L, device=dev),
                position_ids=torch.arange(orig_len, orig_len + L,
                                          device=dev).unsqueeze(0))
    return out.past_key_values, out.logits[:, -1:].clone(), orig_len + L


# Runner.
def load_done(path):
    """
    Resume key. `task` is carried so a RULER run and a synthetic run can share
    one results file without either skipping the other's cells; it is the empty
    string for synthetic rows, which is what they were written with before
    RULER existed, so old files still resume correctly.
    """
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r.get("task", ""), r["method"], r["ratio"],
                              r["depth"], r["trial"],
                              r.get("condition", "qlast")))
                except Exception:
                    pass
    return done


def _transformers_version():
    """Version string of the installed transformers, or why it is unknown."""
    try:
        import transformers
        return transformers.__version__
    except Exception as exc:                     # pragma: no cover
        return "unknown (%s)" % exc


def report_hardware():
    """
    Printed at startup so every log records what it actually ran on. Kaggle's
    script kernels do not let the API ask for a specific accelerator, so the
    only reliable way to know whether a model will fit is to have the run say
    what it was given.
    """
    print("env: python %s" % sys.version.split()[0], flush=True)
    print("env: torch %s  cuda %s  transformers %s"
          % (torch.__version__, torch.version.cuda, _transformers_version()),
          flush=True)
    if not torch.cuda.is_available():
        print("hardware: cpu", flush=True)
        return 0
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print("hardware: cuda:%d %s %.1f GiB"
              % (i, p.name, p.total_memory / 1024 ** 3), flush=True)
    return n


def hf_auth():
    """Make an HF token available to `from_pretrained`, without ever logging it.

    Order of preference: a Kaggle Secret named HF_TOKEN, then the HF_TOKEN or
    HUGGING_FACE_HUB_TOKEN environment variable, then anonymous access.

    Two deliberate choices here. First, the token is exported into the
    environment rather than passed to `from_pretrained(token=...)`: transformers
    and huggingface_hub both read HF_TOKEN on their own, so nothing downstream
    has to carry the value around, and it cannot end up in a traceback frame or
    a repr. Second, only the SOURCE is printed, never the value and never a
    prefix of it -- a kernel log is a public artefact the moment the notebook
    is shared, and a leaked prefix is still a leak.

    Returns True if a token was found. Anonymous is not an error: every model
    used in this study (Qwen2.5, Phi-3.5) is ungated, so a missing token must
    degrade quietly rather than abort a multi-hour run.
    """
    tok = None
    src = "anonymous (no token found; fine for ungated models)"
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            src = "kaggle secret HF_TOKEN"
    except Exception:
        tok = None                      # not on Kaggle, or secret not attached
    if not tok:
        tok = (os.environ.get("HF_TOKEN")
               or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if tok:
            src = "environment"
    if tok:
        tok = tok.strip()
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    print("hf auth: %s" % src, flush=True)
    return bool(tok)


def build_model():
    torch.manual_seed(SEED)
    n_gpu = report_hardware()
    have_token = hf_auth()
    # The tokenizer is the first network call, so a gated repo fails here rather
    # than hours in. The stock error says the repo was not found.
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception as exc:
        msg = str(exc).lower()
        gated = any(w in msg for w in ("gated", "401", "403", "awaiting a review",
                                       "not authorized", "restricted", "access to model"))
        if gated:
            raise SystemExit(
                "\ncannot access %s -- it is gated and no usable token was found "
                "(hf_auth reported: token present = %s).\n"
                "  1. accept the licence at https://huggingface.co/%s\n"
                "  2. on the Kaggle notebook: Add-ons -> Secrets -> add HF_TOKEN, "
                "then ATTACH it to this notebook\n"
                "  3. re-push.\n"
                "Note: a kernel created by `kernels/push` starts with NO secrets "
                "attached, so the first run of a gated model always needs this "
                "second pass. Ungated models (Qwen2.5, Phi-3.5) are unaffected.\n"
                % (MODEL, have_token, MODEL))
        raise
    # Spread across cards only when asked and only when there is more than one.
    # An automatic device map on one card adds accelerate hooks for no benefit
    # and will offload to CPU rather than fail.
    multi = os.environ.get("MULTI_GPU", "") == "1" and n_gpu > 1
    kw = dict(attn_implementation="eager")
    if multi:
        # The cap has to force an even split, not merely fit the weights. The device
        # map fills cards in order, so a cap near a card's capacity puts most of the
        # model on card 0 and leaves it no room for activations. 60% keeps it even.
        per_card = min(torch.cuda.get_device_properties(i).total_memory
                       for i in range(n_gpu)) / 1024 ** 3
        cap = max(1.0, per_card * 0.60)
        # 'balanced' splits evenly by design. 'auto' is an alias for it in current
        # transformers but meant greedy fill in older versions.
        kw.update(device_map="balanced",
                  max_memory={i: "%.1fGiB" % cap for i in range(n_gpu)})
        print("sharding model over %d GPUs, %.1f GiB cap each (card = %.1f GiB)"
              % (n_gpu, cap, per_card), flush=True)
    try:                      # transformers >= 5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, **kw)
    model = (model if multi else model.to(DEVICE)).eval()
    if multi:
        # Refuse a silent offload. A tighter cap makes accelerate more willing to use
        # cpu or disk, which does not fail, it just runs about 50x slower.
        dmap = getattr(model, "hf_device_map", None) or {}
        off = sorted({str(v) for v in dmap.values()
                      if str(v) in ("cpu", "disk")})
        if off:
            raise SystemExit(
                "model was offloaded to %s under max_memory=%.1fGiB per card. "
                "Raise the cap or use a smaller model; a run with offloaded "
                "layers would finish, slowly, and its timings would be "
                "meaningless." % (", ".join(off), cap))
        # Settle the placement question in the log rather than inferring it
        # from an OOM later.
        for i in range(n_gpu):
            print("placed: cuda:%d holds %.2f GiB of weights"
                  % (i, torch.cuda.memory_allocated(i) / 1024 ** 3), flush=True)
        n_layers = len({str(v) for v in dmap.values()}) if dmap else 0
        print("device map spans %d distinct devices" % n_layers, flush=True)
    ok = install_ragged_mask_patch(model)
    print("ragged-mask patch installed: %s" % ok, flush=True)
    if not ok and "pyramidkv" in METHODS:
        print("  WARNING: patch failed; pyramidkv will error on ragged layers",
              flush=True)
    return tok, model


def main_ruler():
    # One driver, two tasksets. RULER and LooGLE differ only in where the samples
    # come from and how a generation is scored, which keeps them comparable.
    loader = loogle_load if TASKSET == "loogle" else ruler_load
    scorer = loogle_score if TASKSET == "loogle" else ruler_score
    label = "loogle" if TASKSET == "loogle" else "ruler"
    # The length label the result rows carry, which is what the table
    # generators' mixed-length guard reads.
    cfg_label = (str(LOOGLE_CTX) if TASKSET == "loogle" else RULER_CONFIG)
    print("model=%s device=%s TASKSET=%s config=%s"
          % (MODEL, DEVICE, label, cfg_label), flush=True)
    tok, model = build_model()

    samples = loader(tok)
    if "qdecoy" in RULER_CONDITIONS:
        attach_decoys(tok, samples)
        n_dec = sum(1 for s in samples if s.get("decoy"))
        print("  decoys attached to %d of %d samples" % (n_dec, len(samples)),
              flush=True)
    tasks = []
    for s in samples:
        if s["task"] not in tasks:
            tasks.append(s["task"])
    print("%s: %d samples over %d task(s) (%s)"
          % (label.upper(), len(samples), len(tasks), ",".join(tasks)), flush=True)
    if label == "loogle":
        print("  %d distinct document(s), truncated to %s tokens"
              % (len({s.get("doc", "") for s in samples}),
                 LOOGLE_CTX or "their natural length"), flush=True)

    done = load_done(RESULTS_PATH)
    print("resuming: %d runs already recorded" % len(done), flush=True)

    # One cell is one (method, ratio) pair: 'full' only at ratio 1.0, every other
    # method only below it.
    cells = [(m, r) for m in METHODS for r in RATIOS
             if (m == "full") == (r == 1.0)]
    # Pass-through cells last, for memory rather than correctness: see note 4.
    cells.sort(key=lambda mr: (mr[0] == "full" or mr[1] >= 1.0))
    groups = [(c, smp) for c in RULER_CONDITIONS for smp in samples]
    if NSHARDS > 1:
        allg = len(groups)
        groups = [g for i, g in enumerate(groups) if i % NSHARDS == SHARD]
        print("shard %d/%d: %d of %d groups"
              % (SHARD, NSHARDS, len(groups), allg), flush=True)
    n_total = len(groups) * len(cells)
    print("grid: %d cells over %d (condition, sample) groups; prefill reuse %s"
          % (n_total, len(groups), "ON" if PREFILL_REUSE else "OFF"), flush=True)

    fh = open(RESULTS_PATH, "a", encoding="utf-8")
    n = 0

    def run_cell(cond, smp, method, ratio, cache, sig, S, suffix_text,
                 pre_secs, shared_by):
        """Compress, answer, score, record. `cache` must hold the master view."""
        t0 = time.time()
        cache, kept = compress(cache, method, S, ratio, sig)
        # Question and answer prefix are forwarded onto the already compressed cache.
        # Under qhidden no policy saw the question while choosing what to keep.
        cache, first_logits, S_eff = extend_with_text(
            model, tok, cache, suffix_text, S)
        text = decode(model, tok, cache, first_logits, orig_len=S_eff,
                      max_new=smp["max_new"])
        score = scorer(smp["task"], text, smp["answers"])
        rec = dict(taskset=label, task=smp["task"], method=method,
                   ratio=ratio, depth=-1.0, trial=smp["index"],
                   condition=cond, ruler_config=cfg_label,
                   decoy_from=smp.get("decoy_from", -1),
                   model=MODEL, ctx=S, seq_len=S, kept=round(kept, 2),
                   actual_ratio=round(kept / S, 4),
                   answer=" | ".join(map(str, smp["answers"]))[:120],
                   output=text.strip()[:160],
                   score=round(score, 4), hit=int(score >= 0.999),
                   secs=round(time.time() - t0, 2),
                   # Prefill is shared across prefill_shared_by cells, so this row's wall-clock is
                   # secs + prefill_secs / prefill_shared_by.
                   prefill_secs=round(pre_secs, 2),
                   prefill_shared_by=shared_by)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("[%5d/%5d] %-8s %-12s r=%.2f %-16s #%d -> %.2f (%.1fs)"
              % (n, n_total, cond, method, ratio, smp["task"],
                 smp["index"], score, rec["secs"]), flush=True)
        del cache, first_logits

    for cond, smp in groups:
        todo = [(m, r) for (m, r) in cells
                if (smp["task"], m, r, -1.0, smp["index"], cond) not in done]
        if not todo:
            n += len(cells)
            continue
        prefix_text, suffix_text = ruler_prompt(tok, smp, cond)
        ids = tok(prefix_text, add_special_tokens=False,
                  return_tensors="pt").input_ids.to(DEVICE)
        S = ids.shape[1]

        if PREFILL_REUSE:
            t_pre = time.time()
            cache, sig, _ = prefill_with_scores(model, ids)
            pre_secs = time.time() - t_pre
            master_kv = [(k, v) for k, v in cache_layers(cache)]
            for method, ratio in todo:
                n += 1
                # Unconditional and first: see note 2.
                rebind_cache(cache, master_kv)
                run_cell(cond, smp, method, ratio, cache, sig, S, suffix_text,
                         pre_secs, len(todo))
            del cache, sig, master_kv
        else:
            for method, ratio in todo:
                n += 1
                t_pre = time.time()
                cache, sig, _ = prefill_with_scores(model, ids)
                pre_secs = time.time() - t_pre
                run_cell(cond, smp, method, ratio, cache, sig, S, suffix_text,
                         pre_secs, 1)
                del cache, sig
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("\nDone. Results at %s" % RESULTS_PATH)


def main():
    if TASKSET in ("ruler", "loogle"):
        return main_ruler()
    print("model=%s device=%s ctx=%d" % (MODEL, DEVICE, CTX), flush=True)
    tok, model = build_model()

    done = load_done(RESULTS_PATH)
    print("resuming: %d runs already recorded" % len(done), flush=True)

    grid = list(itertools.product(CONDITIONS, METHODS, RATIOS, DEPTHS,
                                  range(N_TRIALS)))
    # 'full' only needs ratio 1.0; other methods do not need ratio 1.0
    grid = [g for g in grid
            if (g[1] == "full") == (g[2] == 1.0)]

    fh = open(RESULTS_PATH, "a", encoding="utf-8")
    for n, (cond, method, ratio, depth, trial) in enumerate(grid, 1):
        if ("", method, ratio, depth, trial, cond) in done:
            continue
        rng = random.Random(SEED * 100003 + trial * 1009 + int(depth * 1000))
        prompt, answer, followup = make_sample(tok, CTX, depth, rng,
                                               condition=cond)
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        S = ids.shape[1]

        t0 = time.time()
        cache, sig, first_logits = prefill_with_scores(model, ids)
        S_eff = S

        cache, kept = compress(cache, method, S_eff, ratio, sig)
        if followup is not None:
            # the cache was compressed with a DIFFERENT question visible;
            # serve this one from it without rebuilding
            cache, first_logits, S_eff = extend_with_text(
                model, tok, cache, followup, S_eff)
        text = decode(model, tok, cache, first_logits, orig_len=S_eff)
        hit = int(answer in text)

        rec = dict(taskset="synthetic", task="", method=method, ratio=ratio,
                   depth=depth, trial=trial,
                   condition=cond, distractors=DISTRACTORS,
                   haystack=HAYSTACK,
                   model=MODEL, ctx=CTX, seq_len=S, kept=round(kept, 2),
                   actual_ratio=round(kept / S, 4),
                   answer=answer, output=text.strip()[:120], hit=hit,
                   score=float(hit),
                   secs=round(time.time() - t0, 2))
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("[%4d/%4d] %-8s %-12s r=%.2f d=%.2f t%d -> %s (%.1fs)"
              % (n, len(grid), cond, method, ratio, depth, trial,
                 "HIT " if hit else "miss", rec["secs"]), flush=True)
        del cache, sig, first_logits
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("\nDone. Results at %s" % RESULTS_PATH)


if __name__ == "__main__":
    main()
