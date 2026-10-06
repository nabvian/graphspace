# A shape error is caught while the graph is being built,
# before any data exists, with the expected and actual shapes.
from graphspace import Graph, TensorSpec

graph = Graph("broken")
graph.input("a", TensorSpec((2, 4), dtype="float32"))
graph.input("b", TensorSpec((3, 4), dtype="float32"))
graph.add("a", "b")          # (2, 4) and (3, 4) do not broadcast
