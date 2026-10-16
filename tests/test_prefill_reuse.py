r"""Prove that sharing one prefill across policies changes no result.

    python test_prefill_reuse.py

Runs the RULER path twice over the same cells on CPU with a tiny model and a
truncated context -- once with PREFILL_REUSE=0, the pre-optimisation path that
prefilled once per cell, once with PREFILL_REUSE=1 -- and asserts the two
produce identical generations, scores and retained-entry counts.

Why this test exists. The optimisation reuses a cache object across policies,
restoring the uncompressed key/value tensors between them. That is sound only
because compress(), quantise_cache() and the model's cache update all allocate
new tensors instead of writing the old ones, and because every cell rebinds
before it runs. Each of those is a property of code that can change. A leak
would not raise: it would silently score one policy against another policy's
compressed cache and report a plausible number. Greedy decoding makes the
comparison exact, so any leak shows up as a differing generation.

Why CPU. This costs no GPU quota, which is the point: the speedup can be
trusted before any of the weekly allowance is spent on it. The equivalence
being checked is about cache bookkeeping, not about kernels, so a small model
on CPU exercises the same logic.

Set TEST_MODEL to use a different checkpoint. The default is small enough to
download and run on a laptop.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io, json, os, subprocess, sys, tempfile, collections

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
# Every policy family: position-only, accumulated attention, observation
# window, layer-varying allocation, last-query, key-geometry, value-norm,
# chunked, quantisation, and the uncompressed baseline.
METHODS = ("full,streamingllm,h2o,snapkv,pyramidkv,tova,keydiff,"
           "valueaware,buzz,kivi")
RATIOS = "1.0,0.5,0.2"
TASKS = "niah_single_1,niah_multikey_1,vt"
SAMPLES = "2"
TRUNC = "256"

KEY = ("task", "method", "ratio", "trial", "condition")
# secs and the prefill bookkeeping are expected to differ: that is the point.
COMPARE = ("output", "score", "hit", "kept", "actual_ratio", "ctx", "seq_len")


def run(reuse, out_dir, hooks=1):
    env = dict(os.environ)
    env.update(
        TASKSET="ruler", MODEL=MODEL, DEVICE="cpu",
        RULER_CONFIG="4096", RULER_TASKS=TASKS, RULER_SAMPLES=SAMPLES,
        RULER_TRUNC=TRUNC, RULER_CONDITIONS="qhidden,qvisible",
        METHODS=METHODS, RATIOS=RATIOS,
        ATTN_HOOKS=str(hooks),
        PREFILL_REUSE=str(reuse),
        # RESULTS_PATH is derived from OUTDIR inside the harness, so the runs
        # are separated by giving each its own output directory.
        OUTDIR=os.path.join(out_dir, "reuse%d-hooks%d" % (reuse, hooks)),
        HF_HUB_DISABLE_PROGRESS_BARS="1",
    )
    os.makedirs(env["OUTDIR"], exist_ok=True)
    path = os.path.join(env["OUTDIR"], "kv_bench_results.jsonl")
    if os.path.exists(path):
        os.remove(path)
    print("--- running with PREFILL_REUSE=%d ATTN_HOOKS=%d ---"
          % (reuse, hooks), flush=True)
    p = subprocess.run([sys.executable, os.path.join(HERE, "kv_bench.py")],
                       env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-4000:])
        print(p.stderr[-4000:], file=sys.stderr)
        raise SystemExit("PREFILL_REUSE=%d run failed" % reuse)
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[tuple(r[k] for k in KEY)] = r
    # Prefill sharing must actually have happened, or the test proves nothing.
    shared = {r.get("prefill_shared_by") for r in rows.values()}
    print("    %d cells, prefill_shared_by=%s" % (len(rows), sorted(shared)))
    return rows, p.stdout


def check_decorators():
    """Every @torch.no_grad() must sit immediately above a `def`.

    Inserting a helper function anchored on `def prefill_with_scores` once put
    it BETWEEN that decorator and the function, so the decorator landed on the
    helper and the prefill ran with autograd enabled. Peak prefill memory went
    up 17x and no value changed, so this test passed twice with the regression
    in place. Values are the wrong instrument for that fault; placement is.
    """
    src = io.open(os.path.join(HERE, "kv_bench.py"), encoding="utf-8").read()
    lines = src.splitlines()
    stray = [(i + 1, (lines[i + 1].strip() if i + 1 < len(lines) else "<eof>")[:60])
             for i, l in enumerate(lines)
             if l.strip() == "@torch.no_grad()"
             and not (i + 1 < len(lines) and lines[i + 1].strip().startswith("def "))]
    if stray:
        for ln, nxt in stray:
            print("  kv_bench.py:%d  @torch.no_grad() followed by %r" % (ln, nxt))
        raise SystemExit("FAIL: a @torch.no_grad() does not decorate a function")
    wrapped = [lines[i + 1].strip()[4:].split("(")[0]
               for i, l in enumerate(lines) if l.strip() == "@torch.no_grad()"]
    print("no_grad decorates: %s" % ", ".join(wrapped))
    for required in ("prefill_with_scores", "decode", "extend_with_text",
                     "compress"):
        if required not in wrapped:
            raise SystemExit("FAIL: %s() is not under @torch.no_grad()" % required)


def main():
    check_decorators()
    # The reference is the pre-optimisation harness: one prefill per cell, and
    # the importance signals read off out.attentions after the forward. Each
    # later configuration turns on one optimisation and must match it exactly.
    with tempfile.TemporaryDirectory(prefix="prefill-reuse-") as d:
        ref, _ = run(0, d, hooks=0)
        variants = [
            ("prefill shared", run(1, d, hooks=0)[0]),
            ("prefill shared + attention reduced in-layer", run(1, d, hooks=1)[0]),
            ("attention reduced in-layer only", run(0, d, hooks=1)[0]),
        ]

    if not ref:
        raise SystemExit("the reference run produced no rows")

    failed = False
    for label, got in variants:
        if set(ref) != set(got):
            only_ref = sorted(set(ref) - set(got))[:5]
            only_got = sorted(set(got) - set(ref))[:5]
            print("\n%s: cell sets differ\n  only in reference: %s\n  only here: %s"
                  % (label, only_ref, only_got))
            failed = True
            continue
        diffs = collections.defaultdict(list)
        for key in sorted(ref):
            for field in COMPARE:
                a, b = ref[key].get(field), got[key].get(field)
                if a != b:
                    diffs[field].append((key, a, b))
        print("\n" + "=" * 72)
        print("%s: compared %d cells on %s"
              % (label, len(ref), ", ".join(COMPARE)))
        if diffs:
            failed = True
            for field, items in sorted(diffs.items()):
                print("  %s: %d differing cells" % (field, len(items)))
                for key, a, b in items[:6]:
                    print("     %s\n        reference : %r\n        this run  : %r"
                          % (" ".join(map(str, key)), a, b))
        else:
            print("  identical on every compared field")
    if failed:
        raise SystemExit("\nFAIL: an optimisation changed results")
    new = variants[1][1]

    # A run where every policy silently produced the same text would compare
    # equal while testing nothing, so require that the policies actually
    # diverged somewhere.
    by_cell = collections.defaultdict(set)
    for key, r in new.items():
        by_cell[(key[0], key[3], key[4])].add(r["output"])
    varied = sum(1 for v in by_cell.values() if len(v) > 1)
    print("\nPASS: every configuration matches the reference exactly")
    print("      %d of %d (task, trial, condition) groups show policies "
          "disagreeing, so the cells were not trivially equal"
          % (varied, len(by_cell)))
    if varied == 0:
        raise SystemExit("FAIL: no policy disagreed anywhere -- the comparison "
                         "is vacuous, raise RULER_TRUNC or use a real model")
    shared = {r.get("prefill_shared_by") for r in new.values()}
    if shared == {1}:
        raise SystemExit("FAIL: prefill was never shared, so the optimised "
                         "path was not exercised")


if __name__ == "__main__":
    main()
