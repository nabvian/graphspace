from graphspace import Graph, ResourceContract, TensorSpec

graph = Graph("classifier", ResourceContract.max_memory(16384, deterministic=True))
spec = TensorSpec((2, 4), dtype="float32", role="activation")
graph.input("left", spec)
graph.input("right", spec)
hidden = graph.add("left", "right")
output = graph.relu(hidden)
graph.output(output)

print(graph.memory_plan())
result, record = graph.execute({"left": [1.0] * 8, "right": [-2.0] * 8})
print(result)
print(record)
