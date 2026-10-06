# The README example: two inputs, add, ReLU, under a 64 KiB memory contract.
from graphspace import Graph, ResourceContract, TensorSpec

graph = Graph("classifier", ResourceContract.max_memory(65536, deterministic=True))
spec = TensorSpec((2, 4), dtype="float32", role="activation")
graph.input("left", spec)
graph.input("right", spec)
hidden = graph.add("left", "right")
graph.output(graph.relu(hidden))

values = {"left": [1.0] * 8, "right": [-2.0, 3.0] * 4}
