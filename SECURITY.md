# Security Policy

## Scope

Graphspace is an analysis and execution library. It must not be treated as a sandbox for untrusted Python code.

## Reporting

Do not publish suspected security vulnerabilities in public issues. Report them privately to the project maintainers with:

- affected version;
- reproduction steps;
- expected and actual behavior;
- impact assessment;
- suggested mitigation, if known.

## Security principles

- Never deserialize untrusted objects implicitly.
- Do not execute arbitrary backend code from graph metadata.
- Keep provenance records free of secrets by default.
- Treat backend-reported metadata as untrusted input.
- Validate resource limits before allocating when possible.
- Keep optional integrations isolated from the core package.

## Untrusted inputs

- Set `ResourceContract.max_memory` when input sizes come from untrusted sources. Symbolic dimensions are bound from input sizes, and a `matmul` of two symbolic dimensions grows with their product. Without a contract, nothing limits memory.
- The contract is checked before any allocation, against the estimate in `docs/API.md`.
- No time limit is enforced. The `python` backend and integer `matmul` on the `numpy` backend are slow on large inputs.
- Masked arrays are rejected. Other array subclasses are used as plain arrays.

## Records and failures

- Content digests are opt-in and unsalted SHA-256. Do not publish digests of sensitive low-entropy data.
- Failures report shapes, dtypes, names, and counts, not tensor values.
