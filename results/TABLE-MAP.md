# Which result files produce which table

Every table in the article that reports a measurement, mapped to the result
sets it is computed from and the command that regenerates it. Tables that
summarise the literature rather than a measurement are listed at the end and
have no result files.

Paths are relative to the repository root. All of these run without a GPU.

## Main article

| Table | Result sets | Command |
| --- | --- | --- |
| 6 `tab:benchdepth` | `needle-qwen-3b-8k-qlast` + `needle-qwen-3b-8k-qlast-buzz-rerun` | `python analysis/make_table.py results/needle-qwen-3b-8k-qlast/results.jsonl results/needle-qwen-3b-8k-qlast-buzz-rerun/results.jsonl` |
| 7 `tab:benchcond` | `needle-qwen-3b-8k-qfirst-reuse` | same script, that file |
| 8 `tab:ruler` | `ruler-llama-8b-4k-qhidden-part1`, `-part2` | `python analysis/make_ruler_table.py results/ruler-llama-8b-4k-qhidden-part*/results.jsonl` |
| 9 `tab:rulertask` | as Table 8 | same command |
| 10 `tab:rulerci-llama318b-4k` | `ruler-llama-8b-4k-qhidden-part1`, `-part2`, `ruler-llama-8b-4k-qvisible`, `ruler-llama-8b-4k-qvisible-baseline-only` | `python analysis/bootstrap_ci.py` over those four |
| 11 `tab:gapmatrix` | the nine cells in `analysis/regen_reports.py` | `python analysis/regen_reports.py` then `python analysis/make_gap_matrix.py` |
| 12 `tab:rulerci-llama318b-32k` | `ruler-llama-8b-32k-qhidden`, `ruler-llama-8b-32k-qvisible` | `bootstrap_ci.py` over both |
| 13 `tab:rulermem` | derived from Equation 1 and the retained fractions; no result file | n/a |
| 14 `tab:decoy` | `ruler-llama-3b-4k-qhidden`, `-qvisible`, `-qdecoy` | `python analysis/regen_reports.py decoy3b4k`, then read the two decoy blocks of `reports/ci_report_decoy3b4k.txt` |

The fifty-sample replication discussed in Section 7.6 but not tabled comes from
`ruler-llama-3b-4k-qhidden-b20-n50` and `ruler-llama-3b-4k-qvisible-b20-n50`.

## Online Resource 1

| Table | Result sets |
| --- | --- |
| S4 `tab:rulermodel` | every `ruler-*-qhidden*` set |
| S5 `tab:rulerci-llama318b-8k` | `ruler-llama-8b-8k-qhidden-b20`, `ruler-llama-8b-8k-qhidden-b10-50`, `ruler-llama-8b-8k-qvisible` |
| S6 `tab:rulerci-llama318b-16k` | `ruler-llama-8b-16k-qhidden-b20`, `-b10-50`, `ruler-llama-8b-16k-qvisible` |
| S7 `tab:rulercond` | the qhidden and qvisible sets at every cell |
| S8 `tab:rulerci-qwen253b-4k` | `ruler-qwen-3b-4k-qhidden`, `-qvisible` |
| S9 `tab:rulerci-qwen253b-8k` | `ruler-qwen-3b-8k-qhidden`, `-qvisible` |
| S10 `tab:rulerci-qwen253b-16k` | `ruler-qwen-3b-16k-qhidden`, `-qvisible` |
| S11 `tab:rulerci-llama323b-4k` | `ruler-llama-3b-4k-qhidden`, `-qvisible` (ten samples per task) |
| S12 `tab:loogleci-llama318b-16k` | `loogle-shortdep_qa-16k-n105` |
| S13 `tab:bench`, S14 `tab:benchdepthloose` | as Tables 6 and 7 |

## Tables with no result files

Table 1 (corpus composition) is counted from `references.bib`. Tables 2, 3, 4,
5, 14 and Online Resource 1's Tables S1 to S3, S15 and S16 describe the
surveyed literature and are not computed from any run.

## Sets in this repository that no table uses

Listed so a reader is not left wondering why they are here.

- `loogle-shortdep_qa-16k-n50` — superseded by the 105-document run
- `loogle-longdep_qa-16k-n50` — the long-dependency split discussed in
  Section 7.6's limitations. The uncompressed model is near the floor on it
  (13.8 token F1, 21 of 50 documents at exactly zero), so it bounds the
  harness rather than the policies and no table reports it
- `ruler-llama-3b-8k-*`, `ruler-llama-3b-16k-*` — ran after the write-up
- `needle-qwen-3b-8k-*-rich` — a richer-distractor haystack, ran after the write-up
- `ruler-qwen-7b-4k-qhidden` — kept as a record of a failure, marked unusable in
  `index.json`: the model is bfloat16, the accelerator has no hardware
  bfloat16, and in float16 a large share of its generations are corrupt
