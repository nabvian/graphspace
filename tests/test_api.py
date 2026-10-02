import inspect
import threading
import unittest

import graphspace
from graphspace import ContractViolation, Graph, Node, ResourceContract, TensorSpec


class TestPublicSurface(unittest.TestCase):

    def test_exported_names(self):
        self.assertEqual(set(graphspace.__all__), {
            "__version__", "Analysis", "Basis", "Claim", "Graph", "Node", "TensorSpec", "ResourceContract",
            "ExecutionRecord", "MemoryPlan", "MemoryValue", "Uncertain", "GraphspaceError", "ShapeMismatch",
            "DTypeMismatch", "ResourceLimitExceeded", "UnknownValue", "ContractViolation", "BackendUnavailable",
            "LowConfidence",
        })
        for name in graphspace.__all__:
            self.assertTrue(hasattr(graphspace, name), name)

    def test_stable_signatures(self):
        signatures = {
            Graph.__init__: "(self, name: 'str', resources: 'ResourceContract | None' = None) -> 'None'",
            Graph.execute: "(self, values: 'Mapping[str, Any]', *, backend: 'str' = 'python', digests: 'bool' = False) -> 'tuple[Any, ExecutionRecord]'",
            Graph.add: "(self, left: 'str', right: 'str', *, name: 'str' = 'add') -> 'str'",
            Graph.reshape: "(self, value: 'str', shape: 'tuple[Any, ...]', *, name: 'str' = 'reshape') -> 'str'",
            Graph.validate: "(self, dims: 'Dims | None' = None) -> 'None'",
            Graph.analyze: "(self, dims: 'Dims | None' = None) -> 'Analysis'",
            Graph.memory_plan: "(self, dims: 'Dims | None' = None) -> 'MemoryPlan'",
            Graph.divide: "(self, left: 'str', right: 'str', *, name: 'str' = 'divide') -> 'str'",
            Graph.scale: "(self, value: 'str', factor: 'float', *, name: 'str' = 'scale') -> 'str'",
            Graph.transpose: "(self, value: 'str', axes: 'tuple[int, ...] | None' = None, *, name: 'str' = 'transpose') -> 'str'",
            Graph.softmax: "(self, value: 'str', *, name: 'str' = 'softmax') -> 'str'",
            Graph.layer_norm: "(self, value: 'str', gamma: 'str', beta: 'str', *, eps: 'float' = 1e-05, name: 'str' = 'layer_norm') -> 'str'",
        }
        for function, expected in signatures.items():
            self.assertEqual(str(inspect.signature(function)), expected, function.__qualname__)

    def test_nodes_are_exported_type(self):
        graph = Graph("g")
        graph.input("x", TensorSpec((2,)))
        graph.relu("x")
        self.assertIsInstance(graph.nodes[0], Node)


class TestApiFixes(unittest.TestCase):

    def test_unsupported_layout_rejected(self):
        with self.assertRaises(ContractViolation) as context:
            TensorSpec((2, 2), layout="column_major")
        self.assertEqual(context.exception.expected, ["row_major"])

    def test_resources_must_be_contract(self):
        graph = Graph("g")
        with self.assertRaises(ContractViolation):
            graph.resources = 4096
        with self.assertRaises(ContractViolation):
            Graph("h", resources={"max_memory_bytes": 1})

    def test_concurrent_execution_with_cache_churn(self):
        graph = Graph("threads")
        graph.input("a", TensorSpec(("N",)))
        graph.input("b", TensorSpec(("N",)))
        graph.output(graph.relu(graph.add("a", "b")))
        errors = []

        def worker(offset):
            try:
                for size in range(1, 40):
                    n = (size + offset) % 20 + 1
                    result, _ = graph.execute({"a": [1.0] * n, "b": [2.0] * n})
                    if result != [3.0] * n:
                        errors.append((n, result))
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(offset,)) for offset in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
