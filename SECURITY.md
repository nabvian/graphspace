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
