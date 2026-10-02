# API Stability Policy

## Stable for 0.1

- `Graph`
- `TensorSpec`
- `ResourceContract`
- `ExecutionRecord`
- `Uncertain`
- Exported failure classes
- `Graph.input`, `output`, `add`, `multiply`, `subtract`, `relu`, `reshape`, `matmul`, `validate`, `memory_plan`, and `execute`

## Experimental

- Optional NumPy and PyTorch adapters
- Claim names and the `Basis` vocabulary
- Failure `expected` and `actual` value formats
- `execute(backend=...)` and the NumPy backend
- Memory-plan field details and buffer-reuse rules
- Provenance field details
- Backend extension interfaces

Experimental APIs may change before 1.0. Stable APIs should receive deprecation warnings before removal.
