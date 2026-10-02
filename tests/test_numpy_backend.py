import importlib.util
import math
import tracemalloc
import unittest
import warnings

from graphspace import Graph, ResourceContract, TensorSpec
from graphspace.failures import (
    BackendUnavailable, DTypeMismatch, ResourceLimitExceeded, ShapeMismatch,
)

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np


def binary_graph(operation, spec):
    graph = Graph(operation)
    graph.input("a", spec)
    graph.input("b", spec)
    graph.output(getattr(graph, operation)("a", "b"))
    return graph


class TestBackendSelection(unittest.TestCase):

    def test_unknown_backend_rejected(self):
        with self.assertRaises(BackendUnavailable):
            binary_graph("add", TensorSpec((1,))).execute({"a": [1], "b": [1]}, backend="cuda")

    @unittest.skipIf(HAS_NUMPY, "numpy is installed")
    def test_missing_numpy_is_structured_error(self):
        with self.assertRaises(BackendUnavailable):
            binary_graph("add", TensorSpec((1,))).execute({"a": [1], "b": [1]}, backend="numpy")

    def test_python_relu_propagates_nan(self):
        graph = Graph("relu")
        graph.input("x", TensorSpec((3,)))
        graph.output(graph.relu("x"))
        result, _ = graph.execute({"x": [float("nan"), -1.0, 2.0]})
        self.assertTrue(math.isnan(result[0]))
        self.assertEqual(result[1:], [0.0, 2.0])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestNumpyBackend(unittest.TestCase):

    def test_returns_shaped_array_in_declared_dtype(self):
        result, record = binary_graph("add", TensorSpec((2, 3))).execute(
            {"a": [1.0] * 6, "b": [2.0] * 6}, backend="numpy")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(record.backend, "numpy")
        self.assertEqual(record.claim("backend_version").value, np.__version__)

    def test_matches_python_backend(self):
        graph = Graph("mlp")
        graph.input("x", TensorSpec((2, 3)))
        graph.input("w", TensorSpec((3, 4)))
        graph.input("bias", TensorSpec((2, 4)))
        hidden = graph.add(graph.matmul("x", "w"), "bias")
        graph.output(graph.reshape(graph.relu(hidden), (8,)))
        values = {
            "x": [0.5, -1.0, 2.0, 1.5, 0.25, -0.75],
            "w": [0.1 * i - 0.5 for i in range(12)],
            "bias": [0.1, -0.2, 0.3, -0.4] * 2,
        }
        expected, py_record = graph.execute(values)
        result, np_record = graph.execute(values, backend="numpy")
        self.assertEqual(result.shape, (8,))
        for got, want in zip(result.tolist(), expected):
            self.assertAlmostEqual(got, want, places=5)
        self.assertEqual(py_record.graph_sha256, np_record.graph_sha256)
        self.assertEqual(py_record.peak_memory_bytes, np_record.peak_memory_bytes)

    def test_digests_match_python_backend_for_exact_values(self):
        values = {"a": [1.0, 2.5, -3.0, 0.0], "b": [4.0, 0.5, 1.0, 8.0]}
        graph = binary_graph("multiply", TensorSpec((2, 2)))
        _, py_record = graph.execute(values, digests=True)
        _, np_record = graph.execute(values, backend="numpy", digests=True)
        self.assertEqual(len(py_record.inputs_sha256), 64)
        self.assertEqual(py_record.inputs_sha256, np_record.inputs_sha256)
        self.assertEqual(py_record.output_sha256, np_record.output_sha256)

    def test_int_operations_exact(self):
        graph = Graph("ints")
        graph.input("a", TensorSpec((2, 2), "int64"))
        graph.input("b", TensorSpec((2, 2), "int64"))
        graph.output(graph.relu(graph.subtract(graph.matmul("a", "b"), "b")))
        values = {"a": [1, 2, 3, 4], "b": [5, -6, 7, 8]}
        expected, _ = graph.execute(values)
        result, _ = graph.execute(values, backend="numpy")
        self.assertEqual(result.dtype, np.int64)
        self.assertEqual(result.ravel().tolist(), expected)

    def test_int_overflow_detected_not_wrapped(self):
        cases = [
            ("add", "int32", [2**31 - 1], [1]),
            ("multiply", "int32", [2**16], [2**16]),
            ("subtract", "int64", [-(2**63)], [1]),
            ("multiply", "int64", [2**62], [2]),
        ]
        for operation, dtype, a, b in cases:
            with self.subTest(operation=operation, dtype=dtype):
                graph = binary_graph(operation, TensorSpec((1,), dtype))
                with self.assertRaises(DTypeMismatch):
                    graph.execute({"a": a, "b": b}, backend="numpy")

    def test_int32_matmul_accumulation_overflow_detected(self):
        graph = binary_graph("matmul", TensorSpec((2, 2), "int32"))
        big = 2**30
        with self.assertRaises(DTypeMismatch):
            graph.execute({"a": [big, big, 0, 0], "b": [1, 0, 1, 0]}, backend="numpy")

    def test_accepts_shaped_and_flat_arrays(self):
        graph = binary_graph("add", TensorSpec((2, 2)))
        shaped = np.ones((2, 2), dtype=np.float32)
        flat = np.arange(4, dtype=np.float64)
        result, _ = graph.execute({"a": shaped, "b": flat}, backend="numpy")
        self.assertEqual(result.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_rejects_bad_arrays(self):
        graph = binary_graph("add", TensorSpec((2, 2)))
        ok = np.ones(4, dtype=np.float32)
        bad = [
            (ShapeMismatch, np.ones(5)),
            (ShapeMismatch, np.ones((4, 1))),
            (DTypeMismatch, np.ones(4, dtype=bool)),
            (DTypeMismatch, np.array(["a", "b", "c", "d"])),
            (DTypeMismatch, [True, 1.0, 1.0, 1.0]),
        ]
        for error, value in bad:
            with self.subTest(value=value), self.assertRaises(error):
                graph.execute({"a": value, "b": ok}, backend="numpy")

    def test_masked_arrays_rejected(self):
        graph = binary_graph("add", TensorSpec((4,)))
        masked = np.ma.masked_array(np.ones(4, dtype=np.float32), mask=[0, 0, 1, 0])
        with self.assertRaises(DTypeMismatch) as context:
            graph.execute({"a": masked, "b": np.ones(4, dtype=np.float32)}, backend="numpy")
        self.assertEqual(context.exception.actual, "MaskedArray")

    def test_array_subclasses_return_plain_arrays(self):
        graph = binary_graph("add", TensorSpec((1, 4)))
        matrix = np.matrix([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        result, _ = graph.execute({"a": matrix, "b": np.ones((1, 4), dtype=np.float32)}, backend="numpy")
        self.assertIs(type(result), np.ndarray)

    def test_range_errors_do_not_expose_values(self):
        graph = binary_graph("add", TensorSpec((3,), "int32"))
        with self.assertRaises(DTypeMismatch) as context:
            graph.execute({"a": np.array([5, 2**40, 7]), "b": np.zeros(3, dtype=np.int64)}, backend="numpy")
        self.assertEqual(context.exception.actual, "1 values out of range")
        self.assertNotIn(str(2**40), str(context.exception.to_dict()))

    def test_int_input_range_checked(self):
        graph = binary_graph("add", TensorSpec((1,), "int32"))
        with self.assertRaises(DTypeMismatch):
            graph.execute({"a": np.array([2**40]), "b": np.array([0])}, backend="numpy")
        with self.assertRaises(DTypeMismatch):
            graph.execute({"a": np.array([1.5]), "b": np.array([0])}, backend="numpy")

    def test_symbolic_dims_from_array_size(self):
        graph = binary_graph("add", TensorSpec(("N", 2)))
        graph.resources = ResourceContract.max_memory(graph.memory_plan({"N": 3}).peak_memory_bytes - 1)
        result, _ = graph.execute({"a": np.ones((2, 2)), "b": np.ones(4)}, backend="numpy")
        self.assertEqual(result.shape, (2, 2))
        with self.assertRaises(ResourceLimitExceeded):
            graph.execute({"a": np.ones(6), "b": np.ones(6)}, backend="numpy")

    def test_float_overflow_follows_ieee_without_warnings(self):
        graph = binary_graph("multiply", TensorSpec((2,), "float32"))
        values = {"a": np.array([3e38, 1.0], dtype=np.float32), "b": np.array([10.0, 0.0], dtype=np.float32)}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result, _ = graph.execute(values, backend="numpy")
        expected, _ = graph.execute({"a": [3e38, 1.0], "b": [10.0, 0.0]})
        self.assertTrue(np.isinf(result[0]))
        self.assertEqual(result[1], 0.0)
        self.assertEqual(expected[1], 0.0)

    def test_relu_propagates_nan(self):
        graph = Graph("relu")
        graph.input("x", TensorSpec((3,)))
        graph.output(graph.relu("x"))
        result, _ = graph.execute({"x": np.array([np.nan, -1.0, 2.0])}, backend="numpy")
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1:].tolist(), [0.0, 2.0])

    def test_matching_inputs_are_not_copied(self):
        graph = Graph("view")
        graph.input("x", TensorSpec((4,)))
        graph.output(graph.reshape("x", (2, 2)))
        source = np.zeros(4, dtype=np.float32)
        result, _ = graph.execute({"x": source}, backend="numpy")
        self.assertTrue(np.shares_memory(result, source))

    def test_converted_inputs_are_copied(self):
        graph = Graph("convert")
        graph.input("x", TensorSpec((4,)))
        graph.output(graph.reshape("x", (2, 2)))
        source = np.zeros(4, dtype=np.float64)
        result, _ = graph.execute({"x": source}, backend="numpy")
        self.assertFalse(np.shares_memory(result, source))

    def test_caller_arrays_are_never_written(self):
        graph = Graph("pipeline")
        graph.input("x", TensorSpec((2, 2)))
        graph.input("y", TensorSpec((2, 2)))
        value = graph.relu(graph.reshape("x", (2, 2)))
        for step in range(4):
            value = graph.subtract(value, "y") if step % 2 == 0 else graph.multiply(value, "y")
        graph.output(graph.relu(value))
        x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
        y = np.full((2, 2), 0.5, dtype=np.float32)
        x_before, y_before = x.copy(), y.copy()
        expected, _ = graph.execute({"x": x.ravel().tolist(), "y": y.ravel().tolist()})
        result, _ = graph.execute({"x": x, "y": y}, backend="numpy")
        np.testing.assert_array_equal(x, x_before)
        np.testing.assert_array_equal(y, y_before)
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-6)

    def test_in_place_steps_reuse_one_buffer(self):
        graph = Graph("pipeline")
        graph.input("x", TensorSpec((256, 256)))
        graph.input("y", TensorSpec((256, 256)))
        value = graph.add("x", "y")
        for _ in range(5):
            value = graph.multiply(value, "y")
        graph.output(value)
        x = np.ones((256, 256), dtype=np.float32)
        y = np.full((256, 256), 2.0, dtype=np.float32)
        tracemalloc.start()
        result, _ = graph.execute({"x": x, "y": y}, backend="numpy")
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.assertEqual(result[0, 0], 3.0 * 2.0**5)
        self.assertLess(peak, 2 * x.nbytes)


if __name__ == "__main__":
    unittest.main()
