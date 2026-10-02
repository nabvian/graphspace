# Changelog

## 0.1.0 — Prototype (unreleased)

- Typed graph construction
- Tensor specifications and validation
- Symbolic dimensions bound at execution
- Resource contracts
- Memory planning with buffer reuse, followed by the executor
- Calibrated per-call bookkeeping in the memory estimate, per CPython version
- NumPy backend follows IEEE float rules without warnings
- NumPy 1.22 or newer
- Masked arrays rejected; failures report counts instead of tensor values
- CI actions pinned to commit SHAs with read-only permissions
- Per-graph cache of the execution plan and graph digest
- CPU execution
- Python and NumPy execution backends
- Input value, length, and dtype validation
- Integer range checking
- Provenance and uncertainty values
- SHA-256 digest of the graph; optional digests of inputs and output
- Structured failures with code, graph, node, expected, actual, and remediation
- `Graph.analyze()` and claims labelled by basis
- Add, subtract, multiply, divide, scale, relu, reshape, transpose, matmul, softmax, and layer_norm operations
- NumPy broadcasting for elementwise operations
- Memory estimate reserves NumPy ufunc buffers and row scratch
- Optional NumPy/PyTorch validation adapters
- CLI, tests, and documentation
- Benchmark suite with Python, NumPy, and PyTorch baselines
- Pre-registered benchmark thresholds, calibration, and replication scripts
- Apache-2.0 license
- Typed package marker
- Stable and experimental API defined in `docs/API_STABILITY.md`
