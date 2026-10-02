import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LARGE_SIZES = ("large", "xlarge")
MEMORY_SIZES = ("full", "large", "xlarge")


def source_sha256():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def run_once(seed, output):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "benchmark.py"), "--size", "full", "large", "xlarge",
         "--seed", str(seed), "--output", str(output)],
        check=True, env=env, stdout=subprocess.DEVNULL,
    )
    with open(output) as handle:
        return json.load(handle)


def cells(report):
    for run in report["runs"]:
        for entry in run["workloads"]:
            ours = entry["implementations"]["graphspace_numpy"]
            theirs = entry["implementations"]["numpy"]
            yield run["size"], entry, ours, theirs


def aggregate(reports):
    ratios = {}
    t2_failures, t3_failures = [], []
    for index, report in enumerate(reports):
        for size, entry, ours, theirs in cells(report):
            key = (size, entry["workload"])
            ratios.setdefault(key, []).append(ours["median_ms"] / theirs["median_ms"])
            if size in MEMORY_SIZES and ours["measured_peak_bytes"] > entry["estimated_peak_bytes"]:
                t2_failures.append({"run": index, "size": size, "workload": entry["workload"]})
            if size in MEMORY_SIZES and entry["workload"] == "memory_pipeline" and \
                    ours["measured_peak_bytes"] >= theirs["measured_peak_bytes"]:
                t3_failures.append({"run": index, "size": size})

    t1_cells = []
    for (size, workload), values in sorted(ratios.items()):
        if size not in LARGE_SIZES:
            continue
        t1_cells.append({
            "size": size,
            "workload": workload,
            "median_ratio": statistics.median(values),
            "min_ratio": min(values),
            "max_ratio": max(values),
            "runs_within": sum(value <= 1.10 for value in values),
        })
    single_run_passes = {
        threshold: sum(report["thresholds"][threshold]["status"] == "PASS" for report in reports)
        for threshold in ("T1", "T2", "T3")
    }
    return {
        "T1": {"status": "PASS" if all(c["median_ratio"] <= 1.10 for c in t1_cells) else "FAIL", "cells": t1_cells},
        "T2": {"status": "FAIL" if t2_failures else "PASS", "failures": t2_failures},
        "T3": {"status": "FAIL" if t3_failures else "PASS", "failures": t3_failures},
        "single_run_passes": single_run_passes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replicate the benchmark across independent processes")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--directory", default=str(ROOT / "benchmarks" / "results" / "round3"))
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "results" / "round3-summary.json"))
    args = parser.parse_args(argv)

    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = source_sha256()
    reports = []
    for seed in range(args.runs):
        reports.append(run_once(seed, directory / f"run-{seed}.json"))
        print(f"run {seed + 1}/{args.runs} done")
    if source_sha256() != source:
        raise SystemExit("src/ changed during the replication")

    summary = {
        "source_sha256": source,
        "runs": args.runs,
        "environment": reports[0]["environment"],
        "verdict": aggregate(reports),
    }
    verdict = summary["verdict"]
    print(f"\nsrc sha256 {source}")
    for threshold in ("T1", "T2", "T3"):
        print(f"{threshold} {verdict[threshold]['status']}   single runs passing: {verdict['single_run_passes'][threshold]}/{args.runs}")
    print(f"\n{'size':7} {'workload':16} {'median':>7} {'min':>7} {'max':>7} {'runs<=1.10':>11}")
    for cell in verdict["T1"]["cells"]:
        print(f"{cell['size']:7} {cell['workload']:16} {cell['median_ratio']:7.3f} {cell['min_ratio']:7.3f} "
              f"{cell['max_ratio']:7.3f} {cell['runs_within']:>8}/{args.runs}")
    for threshold in ("T2", "T3"):
        for failure in verdict[threshold]["failures"]:
            print(f"{threshold} failure: {failure}")
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


if __name__ == "__main__":
    main()
