# API Stability Policy

## Stable for 0.1

- `Graph(name, resources=None)` and `Graph.resources`, `inputs`, `nodes`
- `Graph.input`, `output`, `add`, `multiply`, `subtract`, `divide`, `scale`, `relu`, `reshape`, `transpose`, `matmul`, `softmax`, `layer_norm`, `validate`, `analyze`, `memory_plan`, and `execute(values, backend="python", digests=False)`
- `TensorSpec(shape, dtype, layout, role)`, `nbytes`, `nbytes_with`, `concrete_shape`, and `symbols`
- `ResourceContract(max_memory_bytes, deterministic)` and `ResourceContract.max_memory`
- Reading `ExecutionRecord`, `Analysis`, `MemoryPlan`, `MemoryValue`, `Node`, and `Claim` fields; `claim(name)`, `Node.attribute(name)`, and `to_dict()`
- `Uncertain` and `require_confidence`
- `GraphspaceError` and its subclasses, with `code`, `message`, `graph`, `node`, `expected`, `actual`, `remediation`, and `to_dict()`
- `__version__`

## Experimental

- Constructing `ExecutionRecord`, `Analysis`, `MemoryPlan`, `MemoryValue`, `Node`, or `Claim` directly
- Claim names and the `Basis` vocabulary
- Failure `expected` and `actual` value formats
- The NumPy backend and `graphspace.backends`
- Memory-plan buffer-reuse rules and the bookkeeping constants in `graphspace.core`
- Optional NumPy and PyTorch adapters in `graphspace.integrations`
- The `graphspace` command-line output
- Anything not exported from `graphspace`

## Rules

- Stable names keep their meaning within 0.x minor releases unless a deprecation warning ships first.
- New fields are added with defaults after existing fields.
- Experimental APIs may change in any 0.x release.
