import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from replicate import source_sha256

ROOT = Path(__file__).resolve().parent.parent
TIME_LIMIT = 1.10


def run_once(seed, sizes, output):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "models.py"), "--size", *sizes, "--seed", str(seed), "--output", str(output)],
        check=True, env=env, stdout=subprocess.DEVNULL,
    )
    with open(output) as handle:
        return json.load(handle)


def aggregate(reports):
    ratios, failures = {}, {"T0": [], "T2": [], "T3": []}
    single = {"T0": 0, "T1": 0, "T2": 0, "T3": 0}
    for index, report in enumerate(reports):
        run_ok = {"T0": True, "T1": True, "T2": True, "T3": True}
        for run in report["runs"]:
            for entry in run["workloads"]:
                key = (run["size"], entry["workload"])
                ours, theirs = entry["implementations"]["graphspace_numpy"], entry["implementations"]["numpy"]
                ratio = ours["median_ms"] / theirs["median_ms"]
                ratios.setdefault(key, []).append(ratio)
                checks = {
                    "T0": ours["correct"] and theirs["correct"],
                    "T1": ratio <= TIME_LIMIT,
                    "T2": ours["measured_peak_bytes"] <= entry["estimated_peak_bytes"],
                    "T3": ours["measured_peak_bytes"] <= theirs["measured_peak_bytes"],
                }
                for threshold, passed in checks.items():
                    if not passed:
                        run_ok[threshold] = False
                        if threshold in failures:
                            failures[threshold].append({"run": index, "size": run["size"], "workload": entry["workload"]})
        for threshold, passed in run_ok.items():
            single[threshold] += passed

    cells = [
        {"size": size, "workload": workload, "median_ratio": statistics.median(values),
         "min_ratio": min(values), "max_ratio": max(values), "runs_within": sum(value <= TIME_LIMIT for value in values)}
        for (size, workload), values in sorted(ratios.items())
    ]
    verdict = {threshold: {"status": "FAIL" if items else "PASS", "failures": items} for threshold, items in failures.items()}
    verdict["T1"] = {"status": "PASS" if all(cell["median_ratio"] <= TIME_LIMIT for cell in cells) else "FAIL", "cells": cells}
    verdict["single_run_passes"] = single
    return verdict


def memory_table(reports):
    table = {}
    for report in reports:
        for run in report["runs"]:
            for entry in run["workloads"]:
                ours, theirs = entry["implementations"]["graphspace_numpy"], entry["implementations"]["numpy"]
                table[(run["size"], entry["workload"])] = (
                    ours["measured_peak_bytes"], theirs["measured_peak_bytes"], entry["estimated_peak_bytes"],
                )
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description="Round 4: model-block benchmark across independent processes")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--size", nargs="+", default=["model", "model_large"])
    parser.add_argument("--directory", default=str(ROOT / "benchmarks" / "results" / "round4"))
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "results" / "round4-summary.json"))
    args = parser.parse_args(argv)

    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = source_sha256()
    reports = []
    for seed in range(args.runs):
        reports.append(run_once(seed, args.size, directory / f"run-{seed}.json"))
        print(f"run {seed + 1}/{args.runs} done")
    if source_sha256() != source:
        raise SystemExit("src/ changed during the run")

    verdict = aggregate(reports)
    summary = {"source_sha256": source, "runs": args.runs, "sizes": args.size,
               "environment": reports[0]["environment"], "verdict": verdict}
    print(f"\nsrc sha256 {source}")
    for threshold in ("T0", "T1", "T2", "T3"):
        print(f"{threshold} {verdict[threshold]['status']}   single runs passing: {verdict['single_run_passes'][threshold]}/{args.runs}")
    print(f"\n{'size':12} {'workload':16} {'median':>7} {'min':>7} {'max':>7} {'runs<=1.10':>11}")
    for cell in verdict["T1"]["cells"]:
        print(f"{cell['size']:12} {cell['workload']:16} {cell['median_ratio']:7.3f} {cell['min_ratio']:7.3f} "
              f"{cell['max_ratio']:7.3f} {cell['runs_within']:>8}/{args.runs}")
    print(f"\n{'size':12} {'workload':16} {'graphspace':>12} {'numpy':>12} {'estimate':>12}  (bytes, last run)")
    for (size, workload), (ours, theirs, estimate) in sorted(memory_table(reports).items()):
        print(f"{size:12} {workload:16} {ours:12d} {theirs:12d} {estimate:12d}")
    for threshold in ("T0", "T2", "T3"):
        for failure in verdict[threshold]["failures"]:
            print(f"{threshold} failure: {failure}")
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


if __name__ == "__main__":
    main()
