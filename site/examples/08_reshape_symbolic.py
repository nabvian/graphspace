# Reshape must preserve the element count provably, symbolic dimensions included.
from graphspace import Graph, TensorSpec

graph = Graph("reshape")
graph.input("x", TensorSpec(("N", 6), dtype="float32"))
flat = graph.reshape("x", ("N", 2, 3))   # fine: N*6 == N*2*3
graph.output(graph.reshape(flat, (3, "N", 2)))
dims = {"N": 2}
values = {"x": [float(i) for i in range(12)]}
