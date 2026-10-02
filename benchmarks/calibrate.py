import argparse
import json
import math
import platform
import random
import tracemalloc

import numpy as np

from graphspace import Graph, TensorSpec, __version__
from graphspace.core import BOOKKEEPING_BASE_BYTES, BOOKKEEPING_INPUT_BYTES, BOOKKEEPING_NODE_BYTES


OPERATIONS = [
    "add", "multiply", "subtract", "divide", "relu", "reshape", "matmul",
    "scale", "transpose", "softmax", "layer_norm", "bias",
]


def random_graph(rng, index):
    n = rng.choice([4, 16, 64, 128])
    graph = Graph(f"calibration_{index}")
    values = [graph.input(f"in{i}", TensorSpec((n, n))) for i in range(rng.randint(1, 4))]
    vectors = []

    def vector():
        if not vectors or rng.random() < 0.3:
            vectors.append(graph.input(f"vec{len(vectors)}", TensorSpec((n,))))
        return rng.choice(vectors)

    for _ in range(rng.randint(1, 16)):
        operation = rng.choice(OPERATIONS)
        value = rng.choice(values)
        if operation in ("relu", "softmax", "transpose"):
            values.append(getattr(graph, operation)(value))
        elif operation == "reshape":
            values.append(graph.reshape(value, (n, n)))
        elif operation == "scale":
            values.append(graph.scale(value, rng.uniform(0.5, 2.0)))
        elif operation == "layer_norm":
            values.append(graph.layer_norm(value, vector(), vector()))
        elif operation == "bias":
            values.append(graph.add(value, vector()))
        else:
            values.append(getattr(graph, operation)(value, rng.choice(values)))
    graph.output(values[-1])
    return graph, n


def inputs_for(graph, n):
    return {name: np.full(spec.shape, 1.0 / n, dtype=np.float32) for name, spec in graph.inputs.items()}


def tensor_estimate(graph):
    plan = graph.memory_plan()
    return plan.peak_memory_bytes - plan.bookkeeping_bytes


def measure(graph, arrays):
    tracemalloc.start()
    try:
        graph.execute(arrays, backend="numpy")
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure per-call executor bookkeeping")
    parser.add_argument("--graphs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    warmup, n = random_graph(random.Random(0), -1)
    warmup.execute(inputs_for(warmup, n), backend="numpy")
    samples = []
    for index in range(args.graphs):
        graph, n = random_graph(rng, index)
        arrays = inputs_for(graph, n)
        inputs = sum(array.nbytes for array in arrays.values())
        cold = measure(graph, arrays) + inputs - tensor_estimate(graph)
        warm = measure(graph, arrays) + inputs - tensor_estimate(graph)
        samples.append({"inputs": len(graph.inputs), "nodes": len(graph.nodes), "cold": cold, "warm": warm})

    observed = np.array([max(s["cold"], s["warm"]) for s in samples], dtype=float)
    design = np.array([[1.0, s["inputs"], s["nodes"]] for s in samples])
    coefficients = np.maximum(np.linalg.lstsq(design, observed, rcond=None)[0], 0.0)
    shift = float(np.max(observed - design @ coefficients))
    fitted = {
        "base": int(math.ceil((coefficients[0] + shift) / 64) * 64),
        "input": int(math.ceil(coefficients[1] / 8) * 8),
        "node": int(math.ceil(coefficients[2] / 8) * 8),
    }

    allowance = [
        BOOKKEEPING_BASE_BYTES + BOOKKEEPING_INPUT_BYTES * s["inputs"] + BOOKKEEPING_NODE_BYTES * s["nodes"]
        for s in samples
    ]
    covered = sum(max(s["cold"], s["warm"]) <= a for s, a in zip(samples, allowance))
    report = {
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "numpy": np.__version__,
        "graphspace": __version__,
        "seed": args.seed,
        "graphs": args.graphs,
        "constants": {"base": BOOKKEEPING_BASE_BYTES, "input": BOOKKEEPING_INPUT_BYTES, "node": BOOKKEEPING_NODE_BYTES},
        "fitted": fitted,
        "covered": covered,
        "max_cold": max(s["cold"] for s in samples),
        "max_warm": max(s["warm"] for s in samples),
        "samples": samples,
    }
    print(f"{report['python']} numpy {report['numpy']}: {covered}/{args.graphs} graphs within the allowance")
    print(f"max cold overshoot {report['max_cold']} B, max warm overshoot {report['max_warm']} B")
    print(f"fitted upper envelope: base={fitted['base']} input={fitted['input']} node={fitted['node']}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    main()
