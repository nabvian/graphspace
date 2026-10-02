import argparse
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone

from graphspace import Graph, GraphspaceError, TensorSpec, __version__

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


SIZES = {
    "quick": {"rows": 32, "cols": 32, "m": 16, "batch": 4, "inputs": 16, "hidden": 8, "outputs": 4, "depth": 4, "python": True},
    "full": {"rows": 256, "cols": 256, "m": 128, "batch": 32, "inputs": 256, "hidden": 128, "outputs": 10, "depth": 8, "python": True},
    "large": {"rows": 1024, "cols": 1024, "m": 1024, "batch": 1024, "inputs": 1024, "hidden": 1024, "outputs": 1024, "depth": 8, "python": False},
    "xlarge": {"rows": 2048, "cols": 2048, "m": 2048, "batch": 2048, "inputs": 2048, "hidden": 2048, "outputs": 2048, "depth": 8, "python": False},
}

THRESHOLDS = {
    "T1": {"sizes": ("large", "xlarge"), "rule": "graphspace_numpy median <= 1.10 x numpy median, every workload"},
    "T2": {"sizes": ("full", "large", "xlarge"), "rule": "graphspace_numpy measured memory <= estimated peak, every workload"},
    "T3": {"sizes": ("full", "large", "xlarge"), "rule": "memory_pipeline graphspace_numpy measured memory < numpy measured memory"},
}


def matmul_lists(a, b, m, k, n):
    return [sum(a[i * k + p] * b[p * n + j] for p in range(k)) for i in range(m) for j in range(n)]


def relu_list(values):
    return [0.0 if value < 0.0 else value for value in values]


class Workload:
    def __init__(self, name, shapes, build, plain, array):
        self.name = name
        self.shapes = shapes
        self.build = build
        self.plain = plain
        self.array = array


def workloads(size):
    rows, cols, m = size["rows"], size["cols"], size["m"]
    batch, n_in, hidden, n_out, depth = size["batch"], size["inputs"], size["hidden"], size["outputs"], size["depth"]

    def build_add():
        graph = Graph("elementwise_add")
        graph.input("a", TensorSpec((rows, cols)))
        graph.input("b", TensorSpec((rows, cols)))
        graph.output(graph.add("a", "b"))
        return graph

    def build_relu():
        graph = Graph("relu_pipeline")
        graph.input("x", TensorSpec((rows, cols)))
        graph.input("b", TensorSpec((rows, cols)))
        graph.output(graph.relu(graph.subtract(graph.relu(graph.add("x", "b")), "b")))
        return graph

    def build_matmul():
        graph = Graph("matmul")
        graph.input("a", TensorSpec((m, m)))
        graph.input("b", TensorSpec((m, m)))
        graph.output(graph.matmul("a", "b"))
        return graph

    def build_mlp(second_rows=hidden):
        graph = Graph("mlp")
        graph.input("x", TensorSpec((batch, n_in)))
        graph.input("w1", TensorSpec((n_in, hidden)))
        graph.input("b1", TensorSpec((batch, hidden)))
        graph.input("w2", TensorSpec((second_rows, n_out)))
        graph.input("b2", TensorSpec((batch, n_out)))
        hidden_value = graph.relu(graph.add(graph.matmul("x", "w1"), "b1"))
        graph.output(graph.add(graph.matmul(hidden_value, "w2"), "b2"))
        return graph

    def build_memory():
        graph = Graph("memory_pipeline")
        graph.input("x", TensorSpec((rows, cols)))
        graph.input("y", TensorSpec((rows, cols)))
        value = "x"
        for step in range(depth):
            value = graph.add(value, "y") if step % 2 == 0 else graph.multiply(value, "y")
        graph.output(value)
        return graph

    def plain_memory(v):
        value = v["x"]
        for step in range(depth):
            if step % 2 == 0:
                value = [a + b for a, b in zip(value, v["y"])]
            else:
                value = [a * b for a, b in zip(value, v["y"])]
        return value

    def array_memory(v, relu):
        value = v["x"]
        for step in range(depth):
            value = value + v["y"] if step % 2 == 0 else value * v["y"]
        return value

    def plain_mlp(v, second_rows=hidden):
        first = relu_list([a + b for a, b in zip(matmul_lists(v["x"], v["w1"], batch, n_in, hidden), v["b1"])])
        return [a + b for a, b in zip(matmul_lists(first, v["w2"], batch, second_rows, n_out), v["b2"])]

    return [
        Workload(
            "elementwise_add", {"a": (rows, cols), "b": (rows, cols)}, build_add,
            lambda v: [a + b for a, b in zip(v["a"], v["b"])],
            lambda v, relu: v["a"] + v["b"],
        ),
        Workload(
            "relu_pipeline", {"x": (rows, cols), "b": (rows, cols)}, build_relu,
            lambda v: relu_list([a - b for a, b in zip(relu_list([a + b for a, b in zip(v["x"], v["b"])]), v["b"])]),
            lambda v, relu: relu(relu(v["x"] + v["b"]) - v["b"]),
        ),
        Workload(
            "matmul", {"a": (m, m), "b": (m, m)}, build_matmul,
            lambda v: matmul_lists(v["a"], v["b"], m, m, m),
            lambda v, relu: v["a"] @ v["b"],
        ),
        Workload(
            "mlp",
            {"x": (batch, n_in), "w1": (n_in, hidden), "b1": (batch, hidden), "w2": (hidden, n_out), "b2": (batch, n_out)},
            build_mlp, plain_mlp,
            lambda v, relu: relu(v["x"] @ v["w1"] + v["b1"]) @ v["w2"] + v["b2"],
        ),
        Workload("memory_pipeline", {"x": (rows, cols), "y": (rows, cols)}, build_memory, plain_memory, array_memory),
    ], build_mlp, plain_mlp


def make_values(shapes, seed, as_lists):
    if np is None:
        rng = random.Random(seed)
        return {name: [rng.uniform(-1.0, 1.0) for _ in range(math.prod(shape))] for name, shape in shapes.items()}
    rng = np.random.default_rng(seed)
    arrays = {name: rng.uniform(-1.0, 1.0, math.prod(shape)) for name, shape in shapes.items()}
    return {name: array.tolist() for name, array in arrays.items()} if as_lists else arrays


def implementations(workload, values, python):
    graph = workload.build()
    impls = {}
    if python:
        lists = {name: list(data) for name, data in values.items()} if np is None else {
            name: data if isinstance(data, list) else data.tolist() for name, data in values.items()
        }
        impls["plain_python"] = (lambda: workload.plain(lists), lists)
        impls["graphspace_python"] = (lambda: graph.execute(lists)[0], lists)
    if np is not None:
        arrays = {name: np.asarray(data, dtype=np.float32).reshape(workload.shapes[name]) for name, data in values.items()}
        relu = lambda x: np.maximum(x, 0)
        impls["numpy"] = (lambda: workload.array(arrays, relu), arrays)
        impls["graphspace_numpy"] = (lambda: graph.execute(arrays, backend="numpy")[0], arrays)
    if torch is not None:
        tensors = {name: torch.tensor(np.asarray(data) if np is not None else data, dtype=torch.float32).reshape(workload.shapes[name])
                   for name, data in values.items()}
        impls["torch"] = (lambda: workload.array(tensors, torch.relu), tensors)
    return impls


def reference(workload, values, python):
    if python or np is None:
        lists = {name: data if isinstance(data, list) else data.tolist() for name, data in values.items()}
        return workload.plain(lists)
    arrays = {name: np.asarray(data, dtype=np.float64).reshape(workload.shapes[name]) for name, data in values.items()}
    return workload.array(arrays, lambda x: np.maximum(x, 0))


def max_error(result, expected):
    if np is not None:
        result = result.numpy() if torch is not None and isinstance(result, torch.Tensor) else result
        got = np.asarray(result, dtype=np.float64).ravel()
        want = np.asarray(expected, dtype=np.float64).ravel()
        if got.size != want.size:
            return math.inf, 0.0
        return float(np.max(np.abs(got - want), initial=0.0)), float(np.max(np.abs(want), initial=0.0))
    got = result if isinstance(result, list) else result.reshape(-1).tolist()
    if len(got) != len(expected):
        return math.inf, 0.0
    return max((abs(a - b) for a, b in zip(got, expected)), default=0.0), max(map(abs, expected), default=0.0)


def input_bytes(inputs):
    total = 0
    for data in inputs.values():
        if isinstance(data, list):
            total += sys.getsizeof(data) + sum(sys.getsizeof(value) for value in data)
        elif np is not None and isinstance(data, np.ndarray):
            total += data.nbytes
        else:
            total += data.numel() * data.element_size()
    return total


def time_ms(fn, repetitions, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    return statistics.median(samples), ordered[math.ceil(0.95 * len(ordered)) - 1]


def traced_peak(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def run_workload(workload, size, seed, repetitions, warmup):
    python = SIZES[size]["python"]
    values = make_values(workload.shapes, seed, as_lists=python)
    expected = reference(workload, values, python)

    start = time.perf_counter()
    graph = workload.build()
    construction_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    graph.validate()
    validation_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    analysis = graph.analyze()
    analysis_ms = (time.perf_counter() - start) * 1000

    rows = {}
    for name, (fn, inputs) in implementations(workload, values, python).items():
        error, scale = max_error(fn(), expected)
        median, p95 = time_ms(fn, repetitions, warmup)
        measured = None if name == "torch" else input_bytes(inputs) + traced_peak(fn)
        rows[name] = {
            "median_ms": median,
            "p95_ms": p95,
            "measured_peak_bytes": measured,
            "max_abs_error": error,
            "correct": error <= 1e-3 * max(1.0, scale),
        }

    for backend, native in (("python", "plain_python"), ("numpy", "numpy")):
        if f"graphspace_{backend}" in rows and native in rows:
            rows[f"graphspace_{backend}"]["overhead_vs_native_ms"] = (
                rows[f"graphspace_{backend}"]["median_ms"] - rows[native]["median_ms"]
            )

    return {
        "workload": workload.name,
        "shapes": {name: list(shape) for name, shape in workload.shapes.items()},
        "construction_ms": construction_ms,
        "validation_ms": validation_ms,
        "analysis_ms": analysis_ms,
        "estimated_peak_bytes": analysis.claim("peak_memory_bytes").value,
        "estimated_reusable_buffers": analysis.memory_plan.reusable_buffers,
        "implementations": rows,
    }


def failure_detection(seed):
    size = SIZES["full"]
    _, build_mlp, plain_mlp = workloads(size)
    hidden, n_out = size["hidden"], size["outputs"]
    bad_shapes = {
        "x": (size["batch"], size["inputs"]), "w1": (size["inputs"], hidden), "b1": (size["batch"], hidden),
        "w2": (hidden + 1, n_out), "b2": (size["batch"], n_out),
    }
    values = make_values(bad_shapes, seed, as_lists=True)
    report = {}

    def attempt(name, fn):
        start = time.perf_counter()
        try:
            fn()
            outcome, stage = "completed without error", "none"
        except GraphspaceError as error:
            outcome, stage = f"{type(error).__name__} ({error.code})", "construction"
        except Exception as error:
            outcome, stage = type(error).__name__, "execution"
        report[name] = {"stage": stage, "outcome": outcome, "time_to_failure_ms": (time.perf_counter() - start) * 1000}

    attempt("graphspace", lambda: build_mlp(second_rows=hidden + 1))
    attempt("plain_python", lambda: plain_mlp(values))
    if np is not None:
        arrays = {name: np.asarray(data, dtype=np.float32).reshape(bad_shapes[name]) for name, data in values.items()}
        attempt("numpy", lambda: np.maximum(arrays["x"] @ arrays["w1"] + arrays["b1"], 0) @ arrays["w2"] + arrays["b2"])
    if torch is not None:
        tensors = {name: torch.tensor(data).reshape(bad_shapes[name]) for name, data in values.items()}
        attempt("torch", lambda: torch.relu(tensors["x"] @ tensors["w1"] + tensors["b1"]) @ tensors["w2"] + tensors["b2"])
    return {"case": "mlp with w2 rows = hidden + 1", "results": report}


def evaluate(runs):
    by_size = {run["size"]: run for run in runs}
    verdicts = {}
    for threshold, spec in THRESHOLDS.items():
        missing = [size for size in spec["sizes"] if size not in by_size]
        if missing:
            verdicts[threshold] = {"rule": spec["rule"], "status": "NOT EVALUATED", "missing_sizes": missing, "failures": []}
            continue
        failures = []
        for size in spec["sizes"]:
            for entry in by_size[size]["workloads"]:
                rows = entry["implementations"]
                if "graphspace_numpy" not in rows:
                    failures.append({"size": size, "workload": entry["workload"], "reason": "numpy not installed"})
                    continue
                ours, theirs = rows["graphspace_numpy"], rows["numpy"]
                if threshold == "T1" and ours["median_ms"] > 1.10 * theirs["median_ms"]:
                    failures.append({"size": size, "workload": entry["workload"],
                                     "ratio": ours["median_ms"] / theirs["median_ms"]})
                if threshold == "T2" and ours["measured_peak_bytes"] > entry["estimated_peak_bytes"]:
                    failures.append({"size": size, "workload": entry["workload"],
                                     "measured": ours["measured_peak_bytes"], "estimated": entry["estimated_peak_bytes"]})
                if threshold == "T3" and entry["workload"] == "memory_pipeline" and \
                        ours["measured_peak_bytes"] >= theirs["measured_peak_bytes"]:
                    failures.append({"size": size, "workload": entry["workload"],
                                     "measured": ours["measured_peak_bytes"], "numpy": theirs["measured_peak_bytes"]})
        verdicts[threshold] = {"rule": spec["rule"], "status": "FAIL" if failures else "PASS", "failures": failures}
    return verdicts


def cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def environment(args):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "cpu": cpu_model(),
        "graphspace": __version__,
        "numpy": np.__version__ if np is not None else None,
        "torch": torch.__version__ if torch is not None else None,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "sizes": args.size,
        "input_generator": "numpy.random.default_rng uniform(-1, 1)" if np is not None else "random.Random uniform(-1, 1)",
        "memory_method": "input storage + tracemalloc peak during the call; torch is not traced",
    }


def print_summary(report):
    env = report["environment"]
    print(f"graphspace {env['graphspace']} | {env['python']} | numpy {env['numpy']} | {env['cpu']}")
    for run in report["runs"]:
        print(f"\n=== size {run['size']}")
        for entry in run["workloads"]:
            print(f"\n{entry['workload']}  estimated_peak={entry['estimated_peak_bytes']}  "
                  f"construct={entry['construction_ms']:.3f}ms validate={entry['validation_ms']:.3f}ms analyze={entry['analysis_ms']:.3f}ms")
            print(f"  {'implementation':18} {'median_ms':>10} {'p95_ms':>10} {'measured_peak':>14} {'correct':>8}")
            for name, row in entry["implementations"].items():
                measured = "-" if row["measured_peak_bytes"] is None else str(row["measured_peak_bytes"])
                print(f"  {name:18} {row['median_ms']:10.3f} {row['p95_ms']:10.3f} {measured:>14} {str(row['correct']):>8}")
    for name, reason in report["not_supported"].items():
        print(f"\n{name}: not supported ({reason})")
    print(f"\nfailure detection: {report['failure_detection']['case']}")
    for name, row in report["failure_detection"]["results"].items():
        print(f"  {name:18} stage={row['stage']:12} {row['outcome']}  {row['time_to_failure_ms']:.3f}ms")
    print("\nthresholds (docs/BENCHMARK_THRESHOLDS.md)")
    for threshold, verdict in report["thresholds"].items():
        print(f"  {threshold} {verdict['status']:13} {verdict['rule']}")
        for failure in verdict["failures"]:
            print(f"      {failure}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Graphspace benchmark suite")
    parser.add_argument("--size", nargs="+", choices=list(SIZES), default=["full"])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write the JSON report to this path")
    args = parser.parse_args(argv)

    if np is None and any(not SIZES[size]["python"] for size in args.size):
        parser.error("sizes large and xlarge require numpy")
    runs = []
    for size in args.size:
        suite, _, _ = workloads(SIZES[size])
        runs.append({"size": size, "workloads": [
            run_workload(workload, size, args.seed, args.repetitions, args.warmup) for workload in suite
        ]})
    report = {
        "environment": environment(args),
        "runs": runs,
        "not_supported": {"confidence_routing": "graphs have no conditional routing"},
        "failure_detection": failure_detection(args.seed),
        "thresholds": evaluate(runs),
    }
    print_summary(report)
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    main()
