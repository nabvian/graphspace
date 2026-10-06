# Project site (GitHub Pages)

**Live: https://nabvian.github.io/graphspace/**

- **Playground**: edit and run Graphspace code in the browser. The unmodified
  `src/graphspace` package runs in [Pyodide](https://pyodide.org) (CPython on
  WebAssembly), self-hosted under `pyodide/`, so no CDN is involved. The core
  is pure standard library, so no other Python packages are loaded.
  `gs_web.py` reports what the package says: nodes, memory plan, analysis
  claims, the execution record, and structured failures with the stage
  (construction, validation, execution) at which they were caught.
- **Benchmarks**: the round verdicts from `docs/BENCHMARK_THRESHOLDS.md` and
  charts read from `benchmarks/results/` at build time.

| file | role |
|---|---|
| `gs_web.py` | the bridge the playground calls |
| `examples/*.py` | the playground examples; CI runs each one |
| `static/` | page, styles, charts (`app.js`) and the Pyodide worker |
| `build.py` | writes `_site/`; downloads Pyodide once from npm into `.pyodide-cache/` |

```bash
python3 site/build.py
cd _site && python3 -m http.server 8000      # http://localhost:8000
```

`.github/workflows/pages.yml` rebuilds and publishes to the `gh-pages` branch
on every push to `main` that touches the package, results or site.
