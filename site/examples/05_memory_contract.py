# The memory plan is checked against the declared contract at validation,
# before execution: 2 KiB cannot hold this graph's buffers and bookkeeping.
from graphspace import Graph, ResourceContract, TensorSpec

graph = Graph("too_small", ResourceContract.max_memory(2048))
spec = TensorSpec((64, 64), dtype="float32")
graph.input("a", spec)
graph.input("b", spec)
graph.output(graph.relu(graph.multiply("a", "b")))

values = {"a": [1.0] * 4096, "b": [0.5] * 4096}
