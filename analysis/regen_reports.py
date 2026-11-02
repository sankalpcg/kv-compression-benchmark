"""Regenerate reports/ci_report_*.txt from the committed results.

    python analysis/regen_reports.py           # all cells
    python analysis/regen_reports.py 8k qwen8k # named cells only

Each cell is one model at one context length, with both visibility conditions
passed to bootstrap_ci.py in a single invocation so the paired bootstrap can
share one draw of trial indices across them. make_gap_matrix.py reads what this
writes, which is what keeps the matrix from drifting away from the intervals.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORTS = os.path.join(ROOT, "reports")

# stem -> the result directories that make up the cell
CELLS = {
    "4k": ["ruler-llama-8b-4k-qhidden-part1", "ruler-llama-8b-4k-qhidden-part2",
           "ruler-llama-8b-4k-qvisible", "ruler-llama-8b-4k-qvisible-baseline-only"],
    "8k": ["ruler-llama-8b-8k-qhidden-b20", "ruler-llama-8b-8k-qvisible"],
    "16k": ["ruler-llama-8b-16k-qhidden-b20", "ruler-llama-8b-16k-qvisible"],
    "32k": ["ruler-llama-8b-32k-qhidden", "ruler-llama-8b-32k-qvisible"],
    "qwen4k": ["ruler-qwen-3b-4k-qhidden", "ruler-qwen-3b-4k-qvisible"],
    "qwen8k": ["ruler-qwen-3b-8k-qhidden", "ruler-qwen-3b-8k-qvisible"],
    "qwen16k": ["ruler-qwen-3b-16k-qhidden", "ruler-qwen-3b-16k-qvisible"],
    "llama3b4k": ["ruler-llama-3b-4k-qhidden", "ruler-llama-3b-4k-qvisible"],
    "llama3b8k": ["ruler-llama-3b-8k-qhidden", "ruler-llama-3b-8k-qvisible"],
    "llama3b16k": ["ruler-llama-3b-16k-qhidden", "ruler-llama-3b-16k-qvisible"],
    "llama3b4kn50": ["ruler-llama-3b-4k-qhidden-b20-n50",
                     "ruler-llama-3b-4k-qvisible-b20-n50"],
    "loogle16k": ["loogle-shortdep_qa-16k-n50"],
    "loogle16kn105": ["loogle-shortdep_qa-16k-n105"],
}


def main(argv):
    wanted = argv or sorted(CELLS)
    unknown = [w for w in wanted if w not in CELLS]
    if unknown:
        sys.stderr.write("unknown cell(s): %s\n" % ", ".join(unknown))
        return 2
    os.makedirs(REPORTS, exist_ok=True)
    failed = []
    for stem in wanted:
        paths = [os.path.join(ROOT, "results", d, "results.jsonl") for d in CELLS[stem]]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print("%-16s skipped, missing %d input(s)" % (stem, len(missing)))
            continue
        # The contrasts go to stderr and the LaTeX table to stdout; the report
        # keeps both, and make_gap_matrix.py reads the contrast block.
        proc = subprocess.run([sys.executable, os.path.join(HERE, "bootstrap_ci.py")] + paths,
                              capture_output=True, text=True)
        if proc.returncode:
            failed.append(stem)
            print("%-16s FAILED\n%s" % (stem, proc.stderr[-400:]))
            continue
        out = os.path.join(REPORTS, "ci_report_%s.txt" % stem)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(proc.stderr)
            fh.write("\n")
            fh.write(proc.stdout)
        print("%-16s %s" % (stem, os.path.relpath(out, ROOT)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
