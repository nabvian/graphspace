# Graphspace API

## Core

- `Graph(name, resources=None)` creates a graph. `graph.resources` must be a `ResourceContract`. `graph.nodes` returns `Node` values.
- `TensorSpec(shape, dtype, layout, role)` describes a tensor. Dimensions are non-negative ints or non-empty symbolic names such as `"N"`. The only layout is `row_major`.
- `ResourceContract.max_memory(bytes, deterministic=False)` declares limits.
- `graph.input(name, spec)` adds an input. Names must be unique within the graph.
- `graph.add(left, right)`, `graph.multiply(left, right)`, `graph.subtract(left, right)`, and `graph.divide(left, right)` create elementwise operations with NumPy broadcasting. Operands need the same dtype; `role` may differ. A symbolic dimension broadcasts only against itself or 1. `divide` requires a float dtype.
- `graph.scale(value, factor)` multiplies a float tensor by a constant.
- `graph.transpose(value, axes=None)` permutes dimensions; the default reverses them.
- `graph.softmax(value)` normalizes the last axis of a float tensor.
- `graph.layer_norm(value, gamma, beta, eps=1e-5)` normalizes the last axis; `gamma` and `beta` have the shape of that axis.
- `graph.relu(value)` creates ReLU.
- `graph.reshape(value, shape)` changes tensor shape. The element count must be provably equal, including symbolic dimensions.
- `graph.matmul(left, right)` creates rank-2 matrix multiplication.
- `graph.output(value)` marks the graph output.
- `graph.memory_plan(dims=None)` returns liveness, buffer assignment, and peak-memory data. `peak_memory_bytes` includes `bookkeeping_bytes`, the per-call executor allowance `BOOKKEEPING_BASE_BYTES + BOOKKEEPING_INPUT_BYTES × inputs + BOOKKEEPING_NODE_BYTES × nodes`. The constants come from `BOOKKEEPING_BY_VERSION` for the running CPython version, or the largest calibrated values for other versions. Elementwise and row operations reuse a contiguous graph-owned buffer of the output shape whose values die at that step; `reshape` of a contiguous value and `transpose` share their input's buffer; caller inputs are never reused. Each elementwise and row operation reserves one NumPy ufunc buffer of up to 8192 elements per operand, and `softmax` and `layer_norm` reserve two row-sized vectors. `peak_memory_bytes` is `None` while symbolic dimensions are unbound; `dims` binds them.
- `graph.analyze(dims=None)` returns an `Analysis` with the memory plan and claims. It reports and does not raise.
- `graph.validate(dims=None)` checks graph completeness and resource contracts. A memory limit that cannot be verified because of unbound symbolic dimensions raises `ContractViolation`.
- `graph.execute(values, backend="python", digests=False)` returns `(output, ExecutionRecord)`. Symbolic dimensions are bound from input sizes before the resource contract is checked. The executor follows the memory plan and releases each value after its last use. The plan, step list, and graph digest are cached per graph state, resource contract, and dimension binding. `digests=True` adds content digests of the inputs and output.

## Backends

- `python` takes flat row-major sequences and returns a flat list.
- `numpy` takes sequences or ndarrays, flat or in the declared shape, and returns an ndarray in the declared shape and dtype. Arrays already in the declared dtype are used without copying, so an output may share memory with an input. Requires `pip install 'graphspace[numpy]'`.
- Integer results outside the dtype range raise `DTypeMismatch` on both backends.
- Float overflow, division by zero, and invalid operations follow IEEE rules without warnings on both backends.
- An unknown or unavailable backend raises `BackendUnavailable`.

## Execution record

`ExecutionRecord` carries the graph name, backend, runtime, the declared determinism flag, peak memory estimate, UTC timestamp, Python version, and the SHA-256 digest of the graph description. With `digests=True` it also carries SHA-256 digests of the inputs and the output.

Input and output digests hash each tensor as little-endian float64 or int64 bytes, ordered by value name, each preceded by a JSON `[name, dtype, length]` header. Both backends produce the same digest for the same values. The graph digest hashes a canonical JSON description of inputs, nodes, output, and resource contract.

## Claims

`Analysis.claims` and `ExecutionRecord.claims` are `Claim(name, value, basis)` tuples. Look one up with `.claim(name)`.

| Basis | Meaning |
|---|---|
| `declared` | Stated by the user |
| `proven` | Established by graph construction rules |
| `inferred` | Derived from input sizes at execution |
| `estimated` | Computed from a model, not observed |
| `runtime_checked` | Verified during execution |
| `measured` | Observed during execution |
| `backend_reported` | Supplied by the backend |

Analysis claims: `shapes_consistent`, `output_spec`, `dimensions`, `unbound_dimensions`, `peak_memory_bytes`, `bookkeeping_bytes`, `max_memory_bytes`, `within_memory_limit`, `deterministic`.

Execution claims: `dimensions`, `inputs_valid`, `integer_range` (int graphs only), `peak_memory_bytes`, `bookkeeping_bytes`, `max_memory_bytes`, `within_memory_limit`, `deterministic`, `backend_version`, and with `digests=True`, `inputs_sha256` and `output_sha256`.

`ExecutionRecord.to_dict()` returns a JSON-serializable dict.

## Failures

All failures derive from `GraphspaceError`: `ShapeMismatch`, `DTypeMismatch`, `ResourceLimitExceeded`, `UnknownValue`, `ContractViolation`, `LowConfidence`, and `BackendUnavailable`.

Each failure has `code`, `message`, `graph`, `node`, `expected`, `actual`, `remediation`, and `to_dict()`. Fields that do not apply are `None`.

## Uncertainty

`Uncertain(value, confidence, calibrated=False).require_confidence(minimum, *, require_calibrated=False)` returns the value, raises `LowConfidence` below the threshold, or `ContractViolation` when calibration is required and missing.

## Optional backends

NumPy and PyTorch adapters are optional. The core package does not require either dependency. `validate_array` / `validate_tensor` check rank, concrete dimensions, consistent symbolic dimensions, and dtype.
