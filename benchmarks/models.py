import argparse
import json
import math

import numpy as np

from benchmark import environment, input_bytes, time_ms, traced_peak
from graphspace import Graph, TensorSpec

SIZES = {
    "smoke": {"batch": 8, "features": 32, "hidden": 16, "classes": 4, "seq": 16, "dim": 8, "ffn_dim": 16, "ffn_hidden": 32},
    "model": {"batch": 256, "features": 784, "hidden": 512, "classes": 10, "seq": 256, "dim": 64, "ffn_dim": 256, "ffn_hidden": 1024},
    "model_large": {"batch": 2048, "features": 784, "hidden": 512, "classes": 10, "seq": 2048, "dim": 64, "ffn_dim": 512, "ffn_hidden": 2048},
}
EPS = 1e-5


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layer_norm(x, gamma, beta):
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + EPS) * gamma + beta


def mlp_numpy(v):
    h = np.maximum(v["x"] @ v["w1"] + v["b1"], 0)
    h = np.maximum(h @ v["w2"] + v["b2"], 0)
    return softmax(h @ v["w3"] + v["b3"])


def attention_numpy(v):
    q, k, value = v["x"] @ v["wq"], v["x"] @ v["wk"], v["x"] @ v["wv"]
    scores = (q @ k.T) * (1.0 / math.sqrt(v["x"].shape[-1]))
    out = softmax(scores) @ value @ v["wo"] + v["bo"]
    return layer_norm(out + v["x"], v["gamma"], v["beta"])


def ffn_numpy(v):
    h = np.maximum(v["x"] @ v["w1"] + v["b1"], 0)
    return layer_norm(h @ v["w2"] + v["b2"] + v["x"], v["gamma"], v["beta"])


def mlp_graph(size):
    b, f, h, c = size["batch"], size["features"], size["hidden"], size["classes"]
    graph = Graph("mlp")
    shapes = {"x": (b, f), "w1": (f, h), "b1": (h,), "w2": (h, h), "b2": (h,), "w3": (h, c), "b3": (c,)}
    for name, shape in shapes.items():
        graph.input(name, TensorSpec(shape))
    hidden = graph.relu(graph.add(graph.matmul("x", "w1"), "b1"))
    hidden = graph.relu(graph.add(graph.matmul(hidden, "w2"), "b2"))
    graph.output(graph.softmax(graph.add(graph.matmul(hidden, "w3"), "b3")))
    return graph, shapes


def attention_graph(size):
    s, d = size["seq"], size["dim"]
    graph = Graph("attention")
    shapes = {"x": (s, d), "wq": (d, d), "wk": (d, d), "wv": (d, d), "wo": (d, d), "bo": (d,), "gamma": (d,), "beta": (d,)}
    for name, shape in shapes.items():
        graph.input(name, TensorSpec(shape))
    q, k, v = graph.matmul("x", "wq"), graph.matmul("x", "wk"), graph.matmul("x", "wv")
    scores = graph.scale(graph.matmul(q, graph.transpose(k)), 1.0 / math.sqrt(d))
    out = graph.add(graph.matmul(graph.matmul(graph.softmax(scores), v), "wo"), "bo")
    graph.output(graph.layer_norm(graph.add(out, "x"), "gamma", "beta", eps=EPS))
    return graph, shapes


def ffn_graph(size):
    s, d, h = size["seq"], size["ffn_dim"], size["ffn_hidden"]
    graph = Graph("transformer_ffn")
    shapes = {"x": (s, d), "w1": (d, h), "b1": (h,), "w2": (h, d), "b2": (d,), "gamma": (d,), "beta": (d,)}
    for name, shape in shapes.items():
        graph.input(name, TensorSpec(shape))
    hidden = graph.relu(graph.add(graph.matmul("x", "w1"), "b1"))
    out = graph.add(graph.add(graph.matmul(hidden, "w2"), "b2"), "x")
    graph.output(graph.layer_norm(out, "gamma", "beta", eps=EPS))
    return graph, shapes


WORKLOADS = {"mlp": (mlp_graph, mlp_numpy), "attention": (attention_graph, attention_numpy), "transformer_ffn": (ffn_graph, ffn_numpy)}


def make_inputs(shapes, seed):
    rng = np.random.default_rng(seed)
    values = {}
    for name, shape in shapes.items():
        if name == "gamma":
            values[name] = rng.uniform(0.5, 1.5, shape)
        elif len(shape) == 2 and name != "x":
            values[name] = rng.uniform(-1.0, 1.0, shape) / math.sqrt(shape[0])
        else:
            values[name] = rng.uniform(-1.0, 1.0, shape)
    return values


def run_workload(name, size_name, seed, repetitions, warmup):
    build, baseline = WORKLOADS[name]
    graph, shapes = build(SIZES[size_name])
    values = make_inputs(shapes, seed)
    arrays = {key: value.astype(np.float32) for key, value in values.items()}
    expected = baseline(values)
    estimated = graph.analyze().claim("peak_memory_bytes").value

    rows = {}
    for label, fn in (("numpy", lambda: baseline(arrays)), ("graphspace_numpy", lambda: graph.execute(arrays, backend="numpy")[0])):
        result = np.asarray(fn(), dtype=np.float64)
        error = float(np.max(np.abs(result - expected)))
        median, p95 = time_ms(fn, repetitions, warmup)
        rows[label] = {
            "median_ms": median,
            "p95_ms": p95,
            "measured_peak_bytes": input_bytes(arrays) + traced_peak(fn),
            "max_abs_error": error,
            "correct": result.shape == expected.shape and error <= 1e-3 * max(1.0, float(np.max(np.abs(expected)))),
        }
    return {
        "workload": name,
        "shapes": {key: list(shape) for key, shape in shapes.items()},
        "estimated_peak_bytes": estimated,
        "implementations": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Graphspace model-block benchmark")
    parser.add_argument("--size", nargs="+", choices=list(SIZES), default=["model"])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    report = {
        "environment": environment(args),
        "runs": [
            {"size": size, "workloads": [run_workload(name, size, args.seed, args.repetitions, args.warmup) for name in WORKLOADS]}
            for size in args.size
        ],
    }
    for run in report["runs"]:
        print(f"\n=== size {run['size']}")
        for entry in run["workloads"]:
            ours, theirs = entry["implementations"]["graphspace_numpy"], entry["implementations"]["numpy"]
            print(f"{entry['workload']:16} time {ours['median_ms']:9.3f} vs {theirs['median_ms']:9.3f} ms "
                  f"({ours['median_ms'] / theirs['median_ms']:.3f}x)  memory {ours['measured_peak_bytes']} vs "
                  f"{theirs['measured_peak_bytes']}  estimate {entry['estimated_peak_bytes']}  "
                  f"correct {ours['correct']}/{theirs['correct']}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    main()
