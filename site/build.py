"""Build the Graphspace GitHub Pages site into ``_site/``.

    python3 site/build.py

Standard library only. Copies the static page; packs the unmodified
``src/graphspace`` package plus the browser bridge into ``graphspace.zip``;
collects the playground examples; summarises ``benchmarks/results`` into
``data/benchmarks.json``; and self-hosts the Pyodide runtime (downloaded once
from the npm registry and cached in ``.pyodide-cache/``), so the playground
needs no CDN. Graphspace's core uses only the standard library, so no Python
packages beyond Pyodide's own stdlib are loaded.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import re
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = ROOT / "_site"
CACHE = ROOT / ".pyodide-cache"
PYODIDE_VERSION = "0.29.3"
PYODIDE_FILES = ("pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json", "LICENSE")
RESULTS = ROOT / "benchmarks" / "results"


def pyodide_runtime() -> Path:
    target = CACHE / PYODIDE_VERSION
    if all((target / name).exists() for name in PYODIDE_FILES if name != "LICENSE"):
        return target
    url = f"https://registry.npmjs.org/pyodide/-/pyodide-{PYODIDE_VERSION}.tgz"
    print(f"downloading {url}")
    data = urllib.request.urlopen(url, timeout=120).read()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.removeprefix("package/")
            if name in PYODIDE_FILES and member.isfile():
                (target / name).write_bytes(archive.extractfile(member).read())
    return target


def load(path: str) -> dict:
    return json.loads((RESULTS / path).read_text())


def ratio_cells(summary: str) -> list[dict]:
    return load(summary)["verdict"]["T1"]["cells"]


def round_table() -> list[dict]:
    """The results tables in docs/BENCHMARK_THRESHOLDS.md, as written there."""
    text = (ROOT / "docs" / "BENCHMARK_THRESHOLDS.md").read_text()
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[0].isdigit():
            if len(cells) == 7:   # round 1-3 table: Round, Date, Machine, T1, T2, T3, Report
                rows.append({"round": int(cells[0]), "machine": cells[2], "T0": None, "T1": cells[3], "T2": cells[4], "T3": cells[5]})
            else:                 # round 4 table adds T0
                rows.append({"round": int(cells[0]), "machine": cells[2], "T0": cells[3], "T1": cells[4], "T2": cells[5], "T3": cells[6]})
    return rows


def round4_memory() -> list[dict]:
    cells: dict[tuple[str, str], dict] = {}
    for path in sorted((RESULTS / "round4").glob("run-*.json")):
        for run in json.loads(path.read_text())["runs"]:
            for workload in run["workloads"]:
                key = (run["size"], workload["workload"])
                impl = workload["implementations"]
                cell = cells.setdefault(key, {"size": key[0], "workload": key[1], "estimated": workload["estimated_peak_bytes"],
                                              "graphspace_max": 0, "numpy_min": None, "runs": 0, "correct": True})
                cell["graphspace_max"] = max(cell["graphspace_max"], impl["graphspace_numpy"]["measured_peak_bytes"])
                numpy_peak = impl["numpy"]["measured_peak_bytes"]
                cell["numpy_min"] = numpy_peak if cell["numpy_min"] is None else min(cell["numpy_min"], numpy_peak)
                cell["correct"] = cell["correct"] and impl["graphspace_numpy"]["correct"] and impl["numpy"]["correct"]
                cell["runs"] += 1
    return list(cells.values())


def calibration() -> dict:
    versions = []
    for path in sorted((RESULTS / "calibration").glob("*-seed2.json")):
        data = json.loads(path.read_text())
        versions.append({"python": data["python"], "covered": data["covered"], "graphs": data["graphs"], "max_cold": data["max_cold"]})
    held = json.loads((RESULTS / "calibration" / "py3.13-seed2.json").read_text())
    c = held["constants"]
    points = [{"allowance": c["base"] + c["input"] * s["inputs"] + c["node"] * s["nodes"], "cold": s["cold"], "warm": s["warm"],
               "inputs": s["inputs"], "nodes": s["nodes"]} for s in held["samples"]]
    return {"versions": versions, "scatter": {"python": held["python"], "constants": c, "points": points}}


def environment() -> dict:
    env = load("round4-summary.json")["environment"]
    return {k: env[k] for k in ("os", "python", "cpu", "graphspace", "numpy", "repetitions", "warmup", "memory_method")}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SITE / "static", OUT)
    (OUT / ".nojekyll").touch()

    runtime = pyodide_runtime()
    (OUT / "pyodide").mkdir()
    for name in PYODIDE_FILES:
        if (runtime / name).exists():
            shutil.copy2(runtime / name, OUT / "pyodide" / name)
    (OUT / "pyodide" / "NOTICE.txt").write_text(
        f"Pyodide {PYODIDE_VERSION}, https://pyodide.org, redistributed unmodified from the npm package.\n"
        "Licensed under the Mozilla Public License 2.0: https://github.com/pyodide/pyodide/blob/main/LICENSE\n")

    with zipfile.ZipFile(OUT / "graphspace.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "src" / "graphspace").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(ROOT / "src").as_posix())
        archive.write(SITE / "gs_web.py", "gs_web.py")
    version = re.search(r'"([^"]+)"', (ROOT / "src" / "graphspace" / "_version.py").read_text()).group(1)

    (OUT / "data").mkdir()
    examples = []
    for path in sorted((SITE / "examples").glob("*.py")):
        code = path.read_text()
        title = code.splitlines()[0].lstrip("# ").strip()
        examples.append({"id": path.stem, "title": title, "code": code})
    (OUT / "data" / "examples.json").write_text(json.dumps(examples, indent=1))
    benchmarks = {
        "environment": environment(), "rounds": round_table(),
        "round3": ratio_cells("round3-summary.json"), "round4": ratio_cells("round4-summary.json"),
        "round4_memory": round4_memory(), "calibration": calibration(),
    }
    (OUT / "data" / "benchmarks.json").write_text(json.dumps(benchmarks, separators=(",", ":")))
    digest = hashlib.sha256((OUT / "graphspace.zip").read_bytes()).hexdigest()
    (OUT / "data" / "build.json").write_text(json.dumps({"graphspace": version, "pyodide": PYODIDE_VERSION, "package_sha256": digest}))
    print(f"built {OUT.relative_to(ROOT)}/ · graphspace {version} · pyodide {PYODIDE_VERSION}")


if __name__ == "__main__":
    main()
