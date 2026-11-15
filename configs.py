"""Named run configurations.

Each preset is a dict of environment variables that kv_bench.py reads at import
time. run.py sets them and calls the harness; nothing here imports torch, so
`python configs.py` is instant and safe to run anywhere.
"""

MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama-3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama-8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

# Nine compressed policies plus the uncompressed baseline. The order is the one
# the tables use.
METHODS = ("full,streamingllm,h2o,snapkv,pyramidkv,keydiff,"
           "tova,valueaware,buzz,kivi")

# Llama-8B needs both cards of a 2x16GB machine. The 3B models fit on one, and
# splitting a model that fits costs a cross-card transfer for nothing.
MULTI_GPU = {"llama-8b"}

COMMON = dict(CHUNK="256", OBS_WINDOW="32", RECENT="32", SINK="4",
              BUZZ_SEG="16", METHODS=METHODS, ATTN_HOOKS="1", PREFILL_REUSE="1")


def _base(model, device="cuda"):
    cfg = dict(COMMON, MODEL=MODELS[model], DEVICE=device)
    if model in MULTI_GPU:
        cfg["MULTI_GPU"] = "1"
    return cfg


def ruler(model, length, condition, ratios="1.0,0.2", samples=10, repo=None):
    cfg = _base(model)
    cfg.update(TASKSET="ruler", RULER_CONFIG=str(length), RULER_SAMPLES=str(samples),
               RULER_MAX_NEW="128", RATIOS=ratios, RULER_CONDITIONS=condition)
    if repo:
        # The 32K mirror carries one blob field per row instead of three columns.
        cfg.update(RULER_REPO=repo, RULER_SCHEMA="blob")
    return cfg


def loogle(model, task, ctx, samples, ratios="1.0,0.2"):
    cfg = _base(model)
    cfg.update(TASKSET="loogle", LOOGLE_TASK=task, LOOGLE_CTX=str(ctx),
               LOOGLE_SAMPLES=str(samples), LOOGLE_MAX_NEW="64",
               LOOGLE_ONE_PER_DOC="1", RULER_MAX_NEW="128", RATIOS=ratios,
               RULER_CONDITIONS="qhidden,qvisible")
    return cfg


def needle(model, ctx, conditions, ratios, trials, haystack="plain", distractors=4):
    cfg = _base(model)
    cfg.update(CTX=str(ctx), MAX_NEW="24", N_TRIALS=str(trials),
               DISTRACTORS=str(distractors), HAYSTACK=haystack, RATIOS=ratios,
               DEPTHS="0.0,0.25,0.5,0.75,1.0", CONDITIONS=conditions)
    return cfg


BLOB_32K = "SaylorTwift/RULER-32768-llama-3.2-tokenizer"

PRESETS = {
    # The synthetic generator, so the only download is the model itself. A RULER
    # smoke would pull the whole 4,096 configuration to read one row from it.
    # Scores mean nothing at this size; what it proves is that the pipeline runs.
    "smoke": dict(COMMON, MODEL=MODELS["qwen-0.5b"], DEVICE="cpu",
                  CTX="512", MAX_NEW="8", N_TRIALS="1", DISTRACTORS="2",
                  HAYSTACK="plain", METHODS="full,snapkv,h2o",
                  RATIOS="1.0,0.5", DEPTHS="0.0,1.0", CONDITIONS="qlast"),

    # The synthetic needle sweep: accuracy against the position of the answer.
    "needle-depth": needle("qwen-3b", 8192, "qlast", "1.0,0.5,0.3,0.2,0.1", 5),
    "needle-cond": needle("qwen-3b", 8192, "qfirst,reuse", "1.0,0.5,0.2", 3),
    "needle-depth-rich": needle("qwen-3b", 8192, "qlast", "1.0,0.5,0.3,0.2,0.1", 5,
                                haystack="rich"),
    "needle-cond-rich": needle("qwen-3b", 8192, "qfirst,reuse", "1.0,0.5,0.2", 3,
                               haystack="rich"),

    # LooGLE: real documents. longdep is multi-hop over the same kind of text.
    "loogle-short-16k": loogle("llama-8b", "shortdep_qa", 16384, 105),
    "loogle-long-16k": loogle("llama-8b", "longdep_qa", 16384, 50),
}

# RULER grids. Four budgets at 4,096 where the sweep started; two at the longer
# lengths, since 0.2 is the budget every cross-length comparison is read at.
for _m in ("qwen-3b", "llama-3b", "llama-8b"):
    for _c in ("qhidden", "qvisible"):
        PRESETS["ruler-%s-4k-%s" % (_m, _c)] = ruler(_m, 4096, _c, "1.0,0.5,0.2,0.1")
        for _n in (8192, 16384):
            PRESETS["ruler-%s-%dk-%s" % (_m, _n // 1024, _c)] = ruler(_m, _n, _c)

for _c in ("qhidden", "qvisible"):
    PRESETS["ruler-llama-8b-32k-%s" % _c] = ruler("llama-8b", 32768, _c, repo=BLOB_32K)

# Fifty samples per task instead of ten, for the width of the intervals rather
# than their position.
PRESETS["ruler-llama-3b-4k-qhidden-n50"] = ruler("llama-3b", 4096, "qhidden", samples=50)
PRESETS["ruler-llama-3b-4k-qvisible-n50"] = ruler("llama-3b", 4096, "qvisible", samples=50)


def grid_size(cfg):
    """Number of (method, ratio) cells per sample, and the number of samples."""
    methods = [m for m in cfg["METHODS"].split(",") if m]
    ratios = [float(r) for r in cfg["RATIOS"].split(",") if r]
    cells = sum(1 for m in methods for r in ratios if (m == "full") == (r == 1.0))
    if cfg.get("TASKSET") == "ruler":
        tasks = len([t for t in cfg.get("RULER_TASKS", "").split(",") if t]) or 13
        samples = tasks * int(cfg["RULER_SAMPLES"])
    elif cfg.get("TASKSET") == "loogle":
        samples = int(cfg["LOOGLE_SAMPLES"])
    else:
        samples = len(cfg["DEPTHS"].split(",")) * int(cfg["N_TRIALS"])
    conds = cfg.get("RULER_CONDITIONS") or cfg.get("CONDITIONS", "qlast")
    return cells, samples * len(conds.split(","))


if __name__ == "__main__":
    for name in sorted(PRESETS):
        cells, samples = grid_size(PRESETS[name])
        print("%-34s %5d rows  (%d samples x %d cells)"
              % (name, cells * samples, samples, cells))
