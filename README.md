# GraphSpec / Graphspace

Executable Python prototype for typed computational graphs, tensor specifications, resource contracts, memory planning, provenance, uncertainty, and structured failures.

## Run without installation

```bash
PYTHONPATH=src python3 examples/demo.py
```

## Run the CLI

```bash
PYTHONPATH=src python3 -m graphspace.cli
```

After `pip install .` the CLI is also available as `graphspace`.

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The core package uses only the Python standard library. NumPy is optional:

```bash
pip install '.[numpy]'
```

```python
result, record = graph.execute(values, backend="numpy")
```

## Benchmark

```bash
PYTHONPATH=src python3 benchmarks/benchmark.py --output results.json
```

Runs elementwise add, ReLU pipeline, matmul, MLP, and memory pipeline workloads against plain Python, NumPy, and PyTorch when installed. `--size quick full large xlarge` selects sizes; `large` and `xlarge` need NumPy. Results are judged against `docs/BENCHMARK_THRESHOLDS.md`.

```bash
PYTHONPATH=src python3 benchmarks/calibrate.py --seed 1
```

Measures executor bookkeeping on random graphs, fits the allowance, and reports how many graphs it covers.

```bash
python3 benchmarks/replicate.py --runs 10
```

Repeats the benchmark in independent processes and judges the thresholds across runs. Reports construction, validation, and analysis time, median and p95 execution time, estimated and measured peak memory, correctness, and the stage at which a shape error is detected. `--size quick` runs small shapes.

## Limitations

- The `python` backend computes float dtypes in Python `float` precision. The `numpy` backend computes in the declared dtype.
- `peak_memory_bytes` is estimated from declared dtypes, value liveness, buffer reuse, and a per-call bookkeeping allowance calibrated on CPython 3.14.7 with NumPy 2.5.3. It does not cover the one-time cost of loading a backend, inputs converted from another dtype, or the widened temporaries used to check integer overflow.
- `ExecutionRecord.deterministic` is the declared contract.
- The PyTorch adapter validates shape and dtype only.
- Graphs have no conditional routing.

Additional documentation is available in `docs/API.md` and `docs/RELEASE_CHECKLIST.md`.

Project policies are documented in `SECURITY.md`, `CONTRIBUTING.md`, and `docs/API_STABILITY.md`.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.
