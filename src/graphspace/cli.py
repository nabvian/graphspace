from .core import Graph, ResourceContract, TensorSpec


def demo() -> int:
    graph = Graph("demo_add", ResourceContract.max_memory(16384, deterministic=True))
    spec = TensorSpec((2, 4), "float32")
    graph.input("a", spec)
    graph.input("b", spec)
    output = graph.add("a", "b")
    graph.output(output)
    result, record = graph.execute({"a": [1.0] * 8, "b": [2.0] * 8})
    print(f"graph={record.graph_name}")
    print(f"result_elements={len(result)}")
    print(f"peak_memory_bytes={record.peak_memory_bytes}")
    print(f"deterministic={record.deterministic}")
    return 0


def main() -> int:
    return demo()


if __name__ == "__main__":
    raise SystemExit(main())
