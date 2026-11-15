# KV-cache compression benchmark

Nine KV-cache compression policies measured against an uncompressed baseline
under one harness: same model, same budget definition, same prefill, same
decode loop. Published numbers for these methods come from different models at
different lengths on different task mixes, which is why they cannot be read
side by side. Everything here is run on a common footing so they can be.

The repository holds the harness, the analysis scripts, and 64,195 scored
generations. The results are committed, so every table can be regenerated
without a GPU. Some of those runs postdate the write-up and are not in any
published table yet; `results/index.json` records what each set is.

## What is measured

| Policy | Importance signal |
| --- | --- |
| StreamingLLM | position alone: attention sinks plus a recent window |
| H2O | accumulated attention over all query rows |
| SnapKV | attention from a trailing observation window |
| PyramidKV | the same signal, allocated differently per layer |
| TOVA | attention from the single most recent query row |
| KeyDiff | distance of a key from the mean key, no attention at all |
| Value-aware | attention weighted by value norm (Guo et al.) |
| BUZZ | segmented local maxima over intervals of the history |
| KIVI | per-channel and per-token quantisation, nothing evicted |

Every evicting policy gets the same budget definition (a fraction of prefill
cache entries retained), the same four sink tokens and the same 32-token recent
window, so the only quantity that differs between those rows is the importance
signal. KIVI keeps every entry and is given the arithmetically matched
precision instead, `round(16r)` bits against an fp16 baseline, so its row
occupies the same memory as the eviction rows beside it.

Two prompt layouts run across the whole grid. Under `qhidden` the context is
compressed and the question is forwarded onto the compressed cache afterwards,
so no policy sees the question while choosing what to keep. Under `qvisible`
the question sits inside the compressed region. The difference between the two
is what the gap matrix reports.

Benchmarks: RULER (13 tasks, 4,096 to 32,768 tokens), LooGLE (real documents,
token F1), and a synthetic needle sweep that varies the depth of the answer
span, which neither published benchmark exposes at a matched budget.

Models: Qwen2.5-3B, Llama-3.2-3B, Llama-3.1-8B. A Qwen2.5-7B run is included
and marked unusable in `results/index.json`, with the reason.

## Check the machine first

```
pip install -r requirements.txt
python probe.py
```

That reports the interpreter, the packages, the GPUs and their memory, free
disk in the Hugging Face cache, and whether the current directory is writable.
It downloads nothing.

```
python probe.py --smoke
```

That adds an end-to-end run: Qwen2.5-0.5B (about 1 GB), six generations
through prefill, compression and decode, checking that every row lands in the
results file with a non-empty generation. It uses the synthetic generator, so
the model is the only download. Measured at 84 seconds on a laptop CPU. The
scores it prints mean nothing at that size, and a clean exit is the result.

## Running a configuration

```
python run.py --list
python run.py ruler-qwen-3b-4k-qhidden --dry-run
python run.py ruler-qwen-3b-4k-qhidden
```

`--dry-run` resolves the preset, prints the environment and the row count, and
stops before anything is downloaded or loaded. Run it first. The grid for a
13-task RULER sweep at four budgets is 3,640 generations, which is a few hours
on two 16 GB cards, and the dry run is how you find out you asked for the wrong
one before paying for it.

Output appends to `results/<preset>/kv_bench_results.jsonl` as JSON lines, one
object per generation. Completed runs are skipped on restart, so a killed
session resumes where it stopped.

Single values can be overridden without editing `configs.py`:

```
python run.py ruler-qwen-3b-4k-qhidden --set RULER_SAMPLES=2 --set METHODS=full,snapkv
```

## Regenerating the tables

Everything under `analysis/` reads the committed JSON lines and needs no GPU.

```
python analysis/bootstrap_ci.py results/ruler-llama-3b-4k-qhidden/results.jsonl \
                               results/ruler-llama-3b-4k-qvisible/results.jsonl
python analysis/make_ruler_table.py results/ruler-qwen-3b-4k-*/results.jsonl
```

The gap matrix is built from the bootstrap reports rather than from the raw
rows, so that the table and the intervals cannot disagree. Write the reports
first, once:

```
python analysis/regen_reports.py      # writes reports/ci_report_*.txt
python analysis/make_gap_matrix.py > gap_matrix.tex
```

`bootstrap_ci.py` runs a paired cluster bootstrap: 10,000 replicates, the task
as the cluster, the trial as the unit, one draw of trial indices shared across
every cell. The pairing is the point, since trial N names the same document and
the same needle in every cell, so two cells differ only in the policy applied
to identical input. It takes one RULER length per invocation and refuses more,
because trial N at 8,192 tokens and trial N at 16,384 are unrelated samples.

## Results layout

`results/index.json` records the model, task set, nominal length, conditions,
budgets and row count for each set, and carries a note wherever a set needs
one. Directory names are derived from those fields rather than from whatever
the run was called at the time.

Three sets need merging rather than reading alone, and say so in the index:
the 4K Llama-8B qhidden sweep is split across `part1` and `part2` by method,
the 4K qvisible sweep has its uncompressed baseline in a separate directory,
and the BUZZ rows of the synthetic sweep were rerun after the first
implementation kept one token per interval, which fragments multi-token facts.

## Tests

They are scripts, not a pytest suite. Run them one at a time.

```
python tests/test_loogle.py        # offline
python tests/test_ruler_blob.py    # offline, reads tests/fixtures/ruler_rows
python tests/test_ruler_plumbing.py   # downloads a tokenizer
python tests/test_prefill_reuse.py    # downloads Qwen2.5-0.5B, runs on CPU
```

`test_prefill_reuse.py` is the one worth knowing about: it checks that sharing
one prefill across the policies scored on a document produces generations
identical to prefilling once per cell, and that reducing attention inside each
layer produces the same importance signals as reading `out.attentions`. Both
are optimisations that would be invisible if they were wrong.

`test_ruler_blob.py` checks that splitting a blob-mirror row recovers what the
three-field mirror carries, character for character. That equivalence is what
lets the 32,768-token configuration be read at all, since only the blob mirror
has it.

## Hardware notes

The runs behind `results/` were done on two 16 GB T4s. Three things from that
which cost time to learn:

Qwen2.5-7B is a bfloat16 model, a T4 is sm_75 and has no hardware bfloat16, and
in float16 its logits go non-finite. The argmax then returns token id 0 and the
run completes with a large fraction of rows quietly corrupt. `decode()` now
raises on the first non-finite logit, and `probe.py` reports the compute
capability.

Asking the model for `output_attentions` gathers all of a model's attention
tensors onto one card under a device map, which is 8 GiB at 16,384 tokens.
`prefill_with_scores()` reduces each layer's attention inside that layer
instead.

A memory cap near a card's capacity does not force an even split. The device
map fills cards in order, so most of the model lands on card 0 with no room for
activations while card 1 sits half empty.

## Limits

Eviction is applied once, after prefill, so H2O and TOVA are measured as
importance signals rather than as the generation-time algorithms their papers
describe. Retained entries keep the rotary phase assigned during prefill.
Scores are averaged over heads, which puts head-wise allocation methods such as
Ada-KV and DuoAttention outside the comparison. KIVI is implemented as
quantise-and-dequantise rather than with packed low-precision kernels, so its
row measures the accuracy cost of the numeric format and not the speed of a
real implementation. Throughput, latency and memory are not reported at all:
they are hardware-bound, and measuring them on one accelerator and setting the
result beside figures obtained on another is the error this harness exists to
avoid.

## Licence and citation

Apache License 2.0. `NOTICE` records the paper this was built for and the
third-party content redistributed here, which is the cached RULER rows under
`tests/fixtures/` and the reference answers stored alongside the generations in
`results/`.

`CITATION.cff` carries the machine-readable citation. Cite the paper rather
than the repository alone:

> Sankalp Mandlecha and Noshin Sabuwala. Context Compression for Long-Context
> Large Language Models: A Survey and the Query-Visibility Gap. 2026.

One field in `CITATION.cff` is a placeholder until the paper is accepted: the
journal name.
