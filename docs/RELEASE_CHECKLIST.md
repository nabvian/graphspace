# Release Checklist

- [x] Installable package metadata
- [x] Public core API
- [x] Typed validation errors
- [x] Resource contracts
- [x] Memory planning
- [x] CPU execution
- [x] Provenance records
- [x] Uncertainty values
- [x] Claims labelled by basis
- [x] Structured failure fields
- [x] Optional NumPy adapter
- [x] NumPy execution backend
- [x] Optional PyTorch adapter
- [x] Broadcasting add, subtract, multiply, and divide; scale, relu, reshape, transpose, matmul, softmax, and layer_norm
- [x] Standard-library tests
- [x] Benchmark harness
- [x] Baseline benchmark suite
- [x] Local test matrix: Python 3.10–3.14 × no NumPy, oldest NumPy wheel, latest NumPy
- [x] CI matrix: Linux, macOS, and Windows × Python 3.10–3.14, oldest and latest NumPy
- [x] Pre-registered benchmark rounds 1–4, including model blocks
- [ ] Benchmarks on a second machine
- [ ] Benchmarks with trained models and real data
- [x] Published to PyPI (0.1.0)
- [x] Signed release tag `v0.1.0` and verified commits
- [ ] Signed release artifacts (PyPI attestations)
- [x] CI workflow configuration added
- [x] Security policy added
- [x] Security review: static scan and manual review
- [x] Private vulnerability reporting enabled
- [ ] Independent security audit
- [x] Contributor and maintenance guidance added
- [x] API stability policy added
- [x] License (Apache-2.0)
- [x] Edge-case and regression tests
- [x] Version control initialized
- [ ] Community testing and maintenance

The unchecked items need a second machine, trained models, trusted publishing, an outside auditor, or users.
