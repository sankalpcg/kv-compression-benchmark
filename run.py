"""Run one named configuration from configs.py.

    python run.py --list
    python run.py ruler-qwen-3b-4k-qhidden --dry-run
    python run.py ruler-qwen-3b-4k-qhidden --out results/my-run

--dry-run resolves the preset, prints the environment and the grid, and stops
before anything is downloaded or loaded. Any environment variable already set
wins over the preset, so one-off overrides work without editing configs.py.
"""
import argparse
import os
import sys

import configs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("preset", nargs="?", help="name from configs.PRESETS")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved config and grid, run nothing")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one config value, repeatable")
    args = ap.parse_args()

    if args.list or not args.preset:
        for name in sorted(configs.PRESETS):
            cells, samples = configs.grid_size(configs.PRESETS[name])
            print("%-34s %6d rows" % (name, cells * samples))
        return 0

    if args.preset not in configs.PRESETS:
        close = [n for n in configs.PRESETS if args.preset in n]
        sys.stderr.write("unknown preset %r\n" % args.preset)
        if close:
            sys.stderr.write("did you mean: %s\n" % ", ".join(sorted(close)[:5]))
        return 2

    cfg = dict(configs.PRESETS[args.preset])
    for pair in args.set:
        if "=" not in pair:
            sys.stderr.write("--set wants KEY=VALUE, got %r\n" % pair)
            return 2
        k, v = pair.split("=", 1)
        cfg[k] = v

    outdir = args.out or os.path.join("results", args.preset)
    cfg["OUTDIR"] = outdir

    cells, samples = configs.grid_size(cfg)
    print("preset   %s" % args.preset)
    print("rows     %d  (%d samples x %d cells)" % (cells * samples, samples, cells))
    print("output   %s" % os.path.join(outdir, "kv_bench_results.jsonl"))
    for k in sorted(cfg):
        print("  %-18s %s" % (k, cfg[k]))

    if args.dry_run:
        print("\ndry run: nothing was downloaded, loaded or written")
        return 0

    os.makedirs(outdir, exist_ok=True)
    for k, v in cfg.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    import kv_bench
    kv_bench.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
