"""Check that this machine can run the harness, before spending a GPU session on it.

    python probe.py              # environment only, no download
    python probe.py --smoke      # also run the smoke preset end to end on CPU

--smoke downloads Qwen2.5-0.5B (about 1 GB) and one RULER row, runs three
policies through prefill, compression and decode, and checks that rows land in
the results file. It takes a couple of minutes on a laptop and needs no GPU.
The scores it prints are meaningless at that size; a clean exit is the result.
"""
import argparse
import importlib
import os
import shutil
import sys
import tempfile

MIN_PYTHON = (3, 9)
NEEDED = ["torch", "transformers", "datasets", "numpy"]


def ok(label, value):
    print("  ok    %-16s %s" % (label, value))


def bad(label, value):
    print("  FAIL  %-16s %s" % (label, value))


def check_environment():
    failures = []

    print("interpreter")
    v = sys.version_info
    if v[:2] >= MIN_PYTHON:
        ok("python", "%d.%d.%d" % v[:3])
    else:
        bad("python", "%d.%d, need %d.%d or newer" % (v[0], v[1], MIN_PYTHON[0], MIN_PYTHON[1]))
        failures.append("python")

    print("packages")
    mods = {}
    for name in NEEDED:
        try:
            mods[name] = importlib.import_module(name)
            ok(name, getattr(mods[name], "__version__", "unknown"))
        except ImportError as exc:
            bad(name, "%s  (pip install -r requirements.txt)" % exc)
            failures.append(name)

    print("accelerator")
    torch = mods.get("torch")
    if torch is None:
        bad("cuda", "torch missing, cannot check")
    elif not torch.cuda.is_available():
        ok("cuda", "not available, CPU runs only")
    else:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            ok("cuda:%d" % i, "%s, %.1f GiB" % (p.name, p.total_memory / 1024 ** 3))
        # fp16 on a card without hardware bfloat16 is the trap worth naming: a
        # bfloat16 model run in fp16 on sm_75 produces non-finite logits and the
        # run completes with corrupt rows.
        cap = torch.cuda.get_device_capability(0)
        if cap < (8, 0):
            ok("bfloat16", "unsupported on sm_%d%d, use an fp16-safe model" % cap)
        else:
            ok("bfloat16", "supported")

    print("disk")
    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    try:
        free = shutil.disk_usage(os.path.dirname(cache) if not os.path.isdir(cache) else cache).free
        label = "%.1f GiB free" % (free / 1024 ** 3)
        if free < 5 * 1024 ** 3:
            bad("hf cache", label + "  (a 3B model needs about 7 GiB)")
            failures.append("disk")
        else:
            ok("hf cache", label)
    except OSError as exc:
        bad("hf cache", exc)

    try:
        with tempfile.TemporaryFile(dir="."):
            pass
        ok("cwd writable", os.getcwd())
    except OSError as exc:
        bad("cwd writable", exc)
        failures.append("cwd")

    return failures


def run_smoke():
    import configs

    outdir = os.path.join("results", "_probe")
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    cfg = dict(configs.PRESETS["smoke"], OUTDIR=outdir)
    cells, samples = configs.grid_size(cfg)
    print("\nsmoke run: %s, %d rows expected" % (cfg["MODEL"], cells * samples))
    for k, val in cfg.items():
        os.environ[k] = val
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    import kv_bench
    kv_bench.main()

    path = os.path.join(outdir, "kv_bench_results.jsonl")
    import json
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if len(rows) != cells * samples:
        print("\nFAIL  wrote %d rows, expected %d" % (len(rows), cells * samples))
        return ["row count"]
    empty = [r for r in rows if not str(r.get("output", "")).strip()]
    if empty:
        print("\nFAIL  %d of %d rows have an empty generation" % (len(empty), len(rows)))
        return ["empty output"]
    print("\nok    %d rows, every one with a generation -> %s" % (len(rows), path))
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true",
                    help="run the smoke preset end to end on CPU")
    args = ap.parse_args()

    failures = check_environment()
    if failures:
        print("\n%d check(s) failed: %s" % (len(failures), ", ".join(failures)))
        return 1
    if args.smoke:
        failures = run_smoke()
        if failures:
            return 1
    else:
        print("\nenvironment looks usable. Add --smoke to run the pipeline once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
