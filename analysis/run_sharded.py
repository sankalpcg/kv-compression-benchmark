r"""Run one sweep as several workers, one per accelerator.

    python run_sharded.py                      # one worker per visible GPU
    NSHARDS=2 python run_sharded.py            # force the worker count
    DRY_RUN=1 python run_sharded.py            # print the plan, launch nothing

Every environment variable the harness reads is passed through unchanged; this
only adds CUDA_VISIBLE_DEVICES, SHARD, NSHARDS and a per-worker OUTDIR, then
waits for all workers and prints the merge command.

WHEN THIS HELPS, AND WHEN IT DOES NOT.

Kaggle gives a session two T4s with 14.6 GiB each. If the model fits on one
card, two independent workers beat one model spread across both: device_map
="auto" is pipeline parallelism, so at batch=1 exactly one card is computing
at any instant and the other holds weights and waits. Two workers use both.

    Qwen2.5-3B   ~6.2 GB fp16   fits one card    -> shard, expect close to 2x
    Qwen2.5-7B  ~15.2 GB fp16   does not fit     -> do not shard
    Llama-3.1-8B ~16.1 GB fp16  does not fit     -> do not shard

For a model that does not fit, this script refuses to shard rather than
launching workers that will each try to claim both cards and fail partway
through a long run.

A NOTE ON QUOTA. Sharding inside ONE session does not spend quota faster: the
weekly allowance is charged on session wall-clock, so finishing in half the
time costs half the quota. Running two separate sessions concurrently is the
opposite -- it buys wall-clock and charges for both. Check the usage meter
once against a short run rather than trusting either claim.
"""
import os, subprocess, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "kv_bench.py")

# fp16 footprint in GiB, for the refusal below. Approximate on purpose: the
# point is to catch "this cannot fit", not to predict the last 100 MB.
FOOTPRINT = {
    "qwen2.5-0.5b": 1.0, "qwen2.5-1.5b": 3.1, "qwen2.5-3b": 6.2,
    "qwen2.5-7b": 15.2, "qwen2.5-14b": 29.5,
    "meta-llama-3.1-8b": 16.1, "llama-3.1-8b": 16.1, "llama-3.2-3b": 6.5,
}


def footprint(model):
    m = model.lower()
    for k, v in FOOTPRINT.items():
        if k in m:
            return v
    return None


def gpu_info():
    """(count, GiB per card) without importing torch into this process."""
    code = ("import torch,json;"
            "print(json.dumps([torch.cuda.device_count()] + "
            "[torch.cuda.get_device_properties(i).total_memory/1024**3 "
            "for i in range(torch.cuda.device_count())]))")
    try:
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=180)
        data = __import__("json").loads(out.stdout.strip().splitlines()[-1])
        return int(data[0]), (min(data[1:]) if len(data) > 1 else 0.0)
    except Exception as exc:
        print("could not query GPUs (%s); assuming 1" % exc)
        return 1, 0.0


def main():
    model = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    n_gpu, per_card = gpu_info()
    want = int(os.environ.get("NSHARDS", n_gpu or 1))
    need = footprint(model)

    print("model      : %s" % model)
    print("accelerators: %d x %.1f GiB" % (n_gpu, per_card))
    print("fp16 weights: %s GiB" % ("unknown" if need is None else "%.1f" % need))

    if want > 1 and need is not None and per_card and need > per_card * 0.92:
        print("\nREFUSING to shard: %s needs about %.1f GiB and one card has "
              "%.1f GiB, so each worker would have to span both cards and the "
              "two workers would contend for the same memory. Run a single "
              "worker (device_map spreads it) and rely on prefill reuse for "
              "the speedup instead." % (model, need, per_card))
        want = 1
    if want > 1 and need is None:
        print("\nWARNING: unknown footprint for this model. Sharding anyway "
              "because NSHARDS was set, but confirm it fits one card.")

    base_out = os.environ.get("OUTDIR",
                              "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
    plan = []
    for shard in range(want):
        env = dict(os.environ)
        env["SHARD"] = str(shard)
        env["NSHARDS"] = str(want)
        env["OUTDIR"] = os.path.join(base_out, "shard%d" % shard) if want > 1 else base_out
        if want > 1:
            env["CUDA_VISIBLE_DEVICES"] = str(shard % max(n_gpu, 1))
        plan.append(env)
        print("  worker %d -> GPU %s, OUTDIR %s"
              % (shard, env.get("CUDA_VISIBLE_DEVICES", "all"), env["OUTDIR"]))

    if os.environ.get("DRY_RUN"):
        print("\nDRY_RUN set, nothing launched.")
        return 0

    # Only now, so a dry run leaves no directories behind.
    for env in plan:
        os.makedirs(env["OUTDIR"], exist_ok=True)

    procs = []
    for shard, env in enumerate(plan):
        log = open(os.path.join(env["OUTDIR"], "worker.log"), "w",
                   encoding="utf-8")
        procs.append((shard, subprocess.Popen(
            [sys.executable, BENCH], env=env, stdout=log,
            stderr=subprocess.STDOUT), log, env["OUTDIR"]))
        print("launched worker %d (pid %d)" % (shard, procs[-1][1].pid))

    bad = []
    for shard, p, log, out in procs:
        rc = p.wait()
        log.close()
        print("worker %d exited %d" % (shard, rc))
        if rc != 0:
            bad.append((shard, os.path.join(out, "worker.log")))

    if bad:
        print("\nFAILED workers; their logs:")
        for shard, path in bad:
            print("  worker %d: %s" % (shard, path))
        return 1

    files = [os.path.join(e["OUTDIR"], "kv_bench_results.jsonl") for e in plan]
    print("\nAll workers finished. Merge with:\n")
    print("    python make_ruler_table.py \\\n        "
          + " \\\n        ".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
