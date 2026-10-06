# A two-layer MLP with a symbolic batch dimension "N".
# N is bound from the input sizes at execution time.
from graphspace import Graph, ResourceContract, TensorSpec

graph = Graph("mlp", ResourceContract.max_memory(1_000_000))
graph.input("x", TensorSpec(("N", 4), dtype="float32", role="activation"))
graph.input("w1", TensorSpec((4, 3), dtype="float32", role="weight"))
graph.input("b1", TensorSpec((3,), dtype="float32", role="weight"))
graph.input("w2", TensorSpec((3, 2), dtype="float32", role="weight"))

h = graph.relu(graph.add(graph.matmul("x", "w1"), "b1"))
graph.output(graph.softmax(graph.matmul(h, "w2")))

dims = {"N": 2}   # bind N so the memory contract can be checked before running
values = {
    "x":  [0.5, -1.0, 2.0, 0.0,   1.0, 1.0, -0.5, 0.25],
    "w1": [0.1, 0.2, -0.3,  0.4, -0.5, 0.6,  0.7, 0.8, -0.9,  0.2, 0.1, 0.0],
    "b1": [0.0, 0.1, -0.1],
    "w2": [1.0, -1.0,  0.5, 0.5,  -0.25, 0.75],
}
