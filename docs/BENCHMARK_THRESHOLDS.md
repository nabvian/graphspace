# Benchmark Thresholds

Fixed on 2026-10-02, before the first run of the planned-buffer executor.

- Sizes: `full` (256²), `large` (1024²), `xlarge` (2048²). `quick` is a smoke size and is not evaluated.
- Implementation under test: `graphspace_numpy` with default options (no content digests).
- Baseline: ordinary NumPy code for the same workload.
- Measured memory: input storage plus the tracemalloc peak during the call.

| ID | Threshold | Sizes |
|---|---|---|
| T1 | Median time ≤ 1.10 × NumPy median, every workload | large, xlarge |
| T2 | Measured memory ≤ estimated peak, every workload | full, large, xlarge |
| T3 | `memory_pipeline` measured memory < NumPy measured memory | full, large, xlarge |

A threshold fails if any workload at any listed size fails it.

## Rounds

- Round 1: planned-buffer executor, no input copy, content digests opt-in.
- Round 2, fixed on 2026-10-02 before its run: per-graph cache of the execution plan and graph digest; estimated peak includes the executor's per-call bookkeeping. Thresholds unchanged. Bookkeeping constants fitted on 300 random graphs (`benchmarks/calibrate.py --seed 1`) and checked on 300 held-out graphs (`--seed 2`), not on the benchmark workloads.
- Round 3, fixed on 2026-10-02 before its run: replication of round 2 with no change to `src/`. Ten independent processes of `benchmarks/benchmark.py --size full large xlarge`, seeds 0–9, run by `benchmarks/replicate.py`.
  - T1: for each workload and size, the median ratio across the ten runs must be ≤ 1.10.
  - T2 and T3: must hold in every run.
  - The number of single runs passing each threshold is reported but does not decide the round.
  - One machine only; a second machine was not available.

## Results

| Round | Date | Machine | T1 | T2 | T3 | Report |
|---|---|---|---|---|---|---|
| 1 | 2026-10-02 | Apple M5 Pro, CPython 3.14.7, NumPy 2.5.3 | FAIL | FAIL | PASS | `benchmarks/results/2026-10-02-apple-m5-pro.json` |
| 2 | 2026-10-02 | Apple M5 Pro, CPython 3.14.7, NumPy 2.5.3 | PASS | PASS | PASS | `benchmarks/results/2026-10-02-apple-m5-pro-round2.json` |
| 3 | 2026-10-02 | Apple M5 Pro, CPython 3.14.7, NumPy 2.5.3, 10 runs | PASS | PASS | PASS | `benchmarks/results/round3-summary.json` |

## Round 4: model blocks

Fixed on 2026-10-02 and committed before its run. Run by `benchmarks/round4.py`: ten independent processes of `benchmarks/models.py --size model model_large`, seeds 0–9, comparing `graphspace_numpy` with ordinary NumPy code for the same block.

- Workloads: `mlp` (784-512-512-10 classifier with bias vectors, ReLU, and softmax), `attention` (single-head self-attention with output projection, residual, and layer norm), and `transformer_ffn` (feed-forward block with residual and layer norm).
- Sizes: `model` (batch 256, sequence 256) and `model_large` (batch 2048, sequence 2048). `smoke` is not evaluated.
- Inputs are random with scaled weights; no trained weights or real data.

| ID | Threshold |
|---|---|
| T0 | Both implementations match a float64 NumPy reference within 1e-3 of the output scale, in every run |
| T1 | Median time ratio across the ten runs ≤ 1.10, for every workload and size |
| T2 | Measured memory ≤ estimated peak, for every workload, size, and run |
| T3 | Measured memory ≤ NumPy measured memory, for every workload, size, and run |

The number of single runs passing each threshold is reported but does not decide the round. One machine only.

### Round 4 results

| Round | Date | Machine | T0 | T1 | T2 | T3 | Report |
|---|---|---|---|---|---|---|---|
| 4 | 2026-10-02 | Apple M5 Pro, CPython 3.14.7, NumPy 2.5.3, 10 runs | PASS | PASS | PASS | PASS | `benchmarks/results/round4-summary.json` |
