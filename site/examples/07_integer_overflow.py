# Integer results outside the dtype's range raise instead of silently wrapping.
from graphspace import Graph, TensorSpec

graph = Graph("int32_sum")
spec = TensorSpec((4,), dtype="int32")
graph.input("a", spec)
graph.input("b", spec)
graph.output(graph.add("a", "b"))

values = {"a": [100, 20, -5, 2_147_483_647], "b": [27, 1, -3, 1]}   # last pair overflows int32
