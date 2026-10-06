# With a symbolic dimension left unbound, peak memory is unknown,
# so a memory limit cannot be verified: the contract is refused, not assumed.
from graphspace import Graph, ResourceContract, TensorSpec

graph = Graph("unbound", ResourceContract.max_memory(10_000))
graph.input("x", TensorSpec(("N", 8), dtype="float32"))
graph.output(graph.relu("x"))
# Try: dims = {"N": 4}
