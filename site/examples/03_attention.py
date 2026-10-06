# Single-head self-attention scores with a residual and layer norm.
from graphspace import Graph, ResourceContract, TensorSpec

graph = Graph("attention", ResourceContract.max_memory(2_000_000, deterministic=True))
f32 = lambda shape, role="activation": TensorSpec(shape, dtype="float32", role=role)
graph.input("x", f32(("S", 4)))
graph.input("wq", f32((4, 4), "weight"))
graph.input("wk", f32((4, 4), "weight"))
graph.input("wv", f32((4, 4), "weight"))
graph.input("gamma", f32((4,), "weight"))
graph.input("beta", f32((4,), "weight"))

q = graph.matmul("x", "wq")
k = graph.matmul("x", "wk")
v = graph.matmul("x", "wv")
scores = graph.softmax(graph.scale(graph.matmul(q, graph.transpose(k)), 0.5))
attended = graph.matmul(scores, v)
graph.output(graph.layer_norm(graph.add(attended, "x"), "gamma", "beta"))

dims = {"S": 3}
eye = [1.0 if i == j else 0.0 for i in range(4) for j in range(4)]
values = {"x": [0.1 * i for i in range(12)], "wq": eye, "wk": eye, "wv": eye,
          "gamma": [1.0] * 4, "beta": [0.0] * 4}
