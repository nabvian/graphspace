import importlib.util
import math
import random
import tracemalloc
import unittest
import warnings

from graphspace import ContractViolation, DTypeMismatch, Graph, ShapeMismatch, TensorSpec

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np


def binary(operation, left, right, dtype="float32"):
    graph = Graph(operation)
    graph.input("a", TensorSpec(left, dtype))
    graph.input("b", TensorSpec(right, dtype))
    graph.output(getattr(graph, operation)("a", "b"))
    return graph


def attention_block(seq, dim):
    graph = Graph("attention")
    graph.input("x", TensorSpec((seq, dim)))
    for name in ("wq", "wk", "wv", "wo"):
        graph.input(name, TensorSpec((dim, dim)))
    graph.input("gamma", TensorSpec((dim,)))
    graph.input("beta", TensorSpec((dim,)))
    graph.input("bias", TensorSpec((dim,)))
    q = graph.matmul("x", "wq")
    k = graph.matmul("x", "wk")
    v = graph.matmul("x", "wv")
    scores = graph.scale(graph.matmul(q, graph.transpose(k)), 1.0 / math.sqrt(dim))
    context = graph.matmul(graph.softmax(scores), v)
    projected = graph.add(graph.matmul(context, "wo"), "bias")
    graph.output(graph.layer_norm(graph.add(projected, "x"), "gamma", "beta"))
    return graph


def attention_values(seq, dim, seed=0):
    rng = random.Random(seed)
    shapes = {"x": seq * dim, "wq": dim * dim, "wk": dim * dim, "wv": dim * dim, "wo": dim * dim,
              "gamma": dim, "beta": dim, "bias": dim}
    return {name: [rng.uniform(-0.5, 0.5) for _ in range(size)] for name, size in shapes.items()}


class TestBroadcasting(unittest.TestCase):

    def test_bias_vector(self):
        result, _ = binary("add", (2, 3), (3,)).execute({"a": [0.0] * 6, "b": [1.0, 2.0, 3.0]})
        self.assertEqual(result, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

    def test_outer_product_shape(self):
        graph = binary("multiply", (2, 1), (1, 3))
        self.assertEqual(graph.analyze().claim("output_spec").value.shape, (2, 3))
        result, _ = graph.execute({"a": [1.0, 2.0], "b": [1.0, 10.0, 100.0]})
        self.assertEqual(result, [1.0, 10.0, 100.0, 2.0, 20.0, 200.0])

    def test_incompatible_shapes_rejected(self):
        with self.assertRaises(ShapeMismatch):
            binary("add", (2, 3), (2,))
        with self.assertRaises(ShapeMismatch):
            binary("add", ("N", 3), ("M", 3))

    def test_symbolic_broadcast(self):
        graph = binary("subtract", ("N", 3), (3,))
        result, _ = graph.execute({"a": [5.0] * 6, "b": [1.0, 2.0, 3.0]})
        self.assertEqual(result, [4.0, 3.0, 2.0, 4.0, 3.0, 2.0])


class TestDivideAndScale(unittest.TestCase):

    def test_divide_follows_ieee(self):
        result, _ = binary("divide", (5,), (5,)).execute(
            {"a": [6.0, 1.0, -1.0, 0.0, 1.0], "b": [3.0, 0.0, 0.0, 0.0, -0.0]})
        self.assertEqual(result[:3], [2.0, math.inf, -math.inf])
        self.assertTrue(math.isnan(result[3]))
        self.assertEqual(result[4], -math.inf)

    def test_divide_requires_float(self):
        with self.assertRaises(DTypeMismatch):
            binary("divide", (2,), (2,), "int32")

    def test_scale(self):
        graph = Graph("scale")
        graph.input("x", TensorSpec((3,)))
        graph.output(graph.scale("x", 0.5))
        result, _ = graph.execute({"x": [2.0, 4.0, -6.0]})
        self.assertEqual(result, [1.0, 2.0, -3.0])

    def test_scale_validation(self):
        graph = Graph("scale")
        graph.input("x", TensorSpec((3,)))
        graph.input("i", TensorSpec((3,), "int32"))
        with self.assertRaises(ContractViolation):
            graph.scale("x", "2")
        with self.assertRaises(ContractViolation):
            graph.scale("x", True)
        with self.assertRaises(DTypeMismatch):
            graph.scale("i", 2.0)

    def test_attributes_change_graph_digest(self):
        digests = []
        for factor in (0.5, 0.25):
            graph = Graph("scale")
            graph.input("x", TensorSpec((1,)))
            graph.output(graph.scale("x", factor))
            digests.append(graph.execute({"x": [1.0]})[1].graph_sha256)
        self.assertNotEqual(*digests)


class TestTranspose(unittest.TestCase):

    def test_default_reverses_axes(self):
        graph = Graph("t")
        graph.input("x", TensorSpec((2, 3)))
        graph.output(graph.transpose("x"))
        result, _ = graph.execute({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        self.assertEqual(result, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    def test_explicit_axes(self):
        graph = Graph("t")
        graph.input("x", TensorSpec((2, 2, 3)))
        out = graph.transpose("x", (1, 0, 2))
        graph.output(out)
        self.assertEqual(graph.analyze().claim("output_spec").value.shape, (2, 2, 3))
        result, _ = graph.execute({"x": [float(value) for value in range(12)]})
        self.assertEqual(result, [0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0, 9.0, 10.0, 11.0])

    def test_invalid_axes_rejected(self):
        graph = Graph("t")
        graph.input("x", TensorSpec((2, 3)))
        for axes in [(0, 0), (0,), (0, 2), (False, True)]:
            with self.subTest(axes=axes), self.assertRaises(ShapeMismatch):
                graph.transpose("x", axes)


class TestSoftmaxAndLayerNorm(unittest.TestCase):

    def test_softmax_rows(self):
        graph = Graph("s")
        graph.input("x", TensorSpec((2, 3)))
        graph.output(graph.softmax("x"))
        result, _ = graph.execute({"x": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]})
        self.assertAlmostEqual(sum(result[:3]), 1.0)
        self.assertEqual(result[3:], [1 / 3] * 3)
        expected = [math.exp(v) / sum(math.exp(u) for u in (1.0, 2.0, 3.0)) for v in (1.0, 2.0, 3.0)]
        for got, want in zip(result[:3], expected):
            self.assertAlmostEqual(got, want)

    def test_softmax_nan_row(self):
        graph = Graph("s")
        graph.input("x", TensorSpec((2, 2)))
        graph.output(graph.softmax("x"))
        result, _ = graph.execute({"x": [math.nan, 1.0, 1.0, 1.0]})
        self.assertTrue(all(math.isnan(value) for value in result[:2]))
        self.assertEqual(result[2:], [0.5, 0.5])

    def test_softmax_validation(self):
        graph = Graph("s")
        graph.input("i", TensorSpec((2,), "int32"))
        graph.input("scalar", TensorSpec(()))
        with self.assertRaises(DTypeMismatch):
            graph.softmax("i")
        with self.assertRaises(ShapeMismatch):
            graph.softmax("scalar")

    def test_layer_norm(self):
        graph = Graph("ln")
        graph.input("x", TensorSpec((2, 4)))
        graph.input("gamma", TensorSpec((4,)))
        graph.input("beta", TensorSpec((4,)))
        graph.output(graph.layer_norm("x", "gamma", "beta", eps=0.0001))
        result, _ = graph.execute({"x": [1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0], "gamma": [1.0] * 4, "beta": [0.0] * 4})
        row = result[:4]
        self.assertAlmostEqual(sum(row), 0.0)
        self.assertAlmostEqual(sum(value * value for value in row) / 4, 1.25 / (1.25 + 0.0001))
        self.assertEqual(result[4:], [0.0] * 4)

    def test_layer_norm_validation(self):
        graph = Graph("ln")
        graph.input("x", TensorSpec((2, 4)))
        graph.input("good", TensorSpec((4,)))
        graph.input("wrong", TensorSpec((3,)))
        graph.input("half", TensorSpec((4,), "float16"))
        with self.assertRaises(ShapeMismatch):
            graph.layer_norm("x", "wrong", "good")
        with self.assertRaises(DTypeMismatch):
            graph.layer_norm("x", "good", "half")
        with self.assertRaises(ContractViolation):
            graph.layer_norm("x", "good", "good", eps=0)


class TestPlanningForNewOperations(unittest.TestCase):

    def test_transpose_aliases_and_reshape_of_transpose_copies(self):
        graph = Graph("t")
        graph.input("x", TensorSpec((2, 3)))
        transposed = graph.transpose("x")
        graph.output(graph.reshape(transposed, (6,)))
        buffers = {value.name: value.buffer for value in graph.memory_plan().values}
        self.assertEqual(buffers, {"x": "x", "transpose": "x", "reshape": "reshape"})

    def test_softmax_runs_in_place_and_counts_row_scratch(self):
        graph = Graph("s")
        graph.input("x", TensorSpec((8, 16)))
        graph.output(graph.softmax(graph.scale("x", 2.0)))
        plan = graph.memory_plan()
        buffers = {value.name: value.buffer for value in plan.values}
        self.assertEqual(buffers["softmax"], "scale")
        self.assertEqual(plan.peak_memory_bytes - plan.bookkeeping_bytes, 2 * 8 * 16 * 4 + 2 * 8 * 4 + 2 * 8 * 16 * 4)

    def test_broadcast_output_never_reuses_smaller_input(self):
        graph = Graph("b")
        graph.input("x", TensorSpec((4, 3)))
        graph.input("v", TensorSpec((3,)))
        vector = graph.scale("v", 2.0)
        graph.output(graph.add(vector, "x"))
        buffers = {value.name: value.buffer for value in graph.memory_plan().values}
        self.assertEqual(buffers["add"], "add")

    def test_attention_block_runs(self):
        result, record = attention_block(4, 8).execute(attention_values(4, 8))
        self.assertEqual(len(result), 32)
        self.assertTrue(all(math.isfinite(value) for value in result))
        self.assertGreater(record.peak_memory_bytes, 0)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestNewOperationsOnNumpy(unittest.TestCase):

    def test_attention_block_matches_python(self):
        graph = attention_block(16, 8)
        values = attention_values(16, 8)
        expected, _ = graph.execute(values)
        result, _ = graph.execute(values, backend="numpy")
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-4, atol=1e-5)

    def test_operations_match_python(self):
        rng = random.Random(3)
        cases = [
            ("divide", (3, 4), (4,)),
            ("multiply", (3, 1), (1, 4)),
            ("subtract", (4,), (3, 4)),
        ]
        for operation, left, right in cases:
            with self.subTest(operation=operation):
                graph = binary(operation, left, right)
                values = {"a": [rng.uniform(0.5, 2.0) for _ in range(math.prod(left))],
                          "b": [rng.uniform(0.5, 2.0) for _ in range(math.prod(right))]}
                expected, _ = graph.execute(values)
                result, _ = graph.execute(values, backend="numpy")
                np.testing.assert_allclose(result.ravel(), expected, rtol=1e-6)

    def test_transpose_returns_declared_shape(self):
        graph = Graph("t")
        graph.input("x", TensorSpec((2, 3, 4)))
        graph.output(graph.transpose("x", (2, 0, 1)))
        values = [float(value) for value in range(24)]
        expected, _ = graph.execute({"x": values})
        result, _ = graph.execute({"x": values}, backend="numpy")
        self.assertEqual(result.shape, (4, 2, 3))
        self.assertEqual(result.ravel().tolist(), expected)

    def test_ieee_division_without_warnings(self):
        graph = binary("divide", (4,), (4,))
        values = {"a": [1.0, -1.0, 0.0, 1.0], "b": [0.0, 0.0, 0.0, -0.0]}
        expected, _ = graph.execute(values)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result, _ = graph.execute(values, backend="numpy")
        self.assertEqual(result[:2].tolist(), expected[:2])
        self.assertTrue(np.isnan(result[2]) and math.isnan(expected[2]))
        self.assertEqual(result[3], -np.inf)

    def test_caller_arrays_unchanged(self):
        graph = attention_block(8, 4)
        arrays = {name: np.asarray(data, dtype=np.float32) for name, data in attention_values(8, 4).items()}
        before = {name: array.copy() for name, array in arrays.items()}
        graph.execute(arrays, backend="numpy")
        for name, array in arrays.items():
            np.testing.assert_array_equal(array, before[name])

    def test_measured_memory_within_estimate(self):
        graph = attention_block(128, 64)
        arrays = {name: np.asarray(data, dtype=np.float32).reshape(spec.shape)
                  for (name, data), spec in zip(attention_values(128, 64).items(), graph.inputs.values())}
        graph.execute(arrays, backend="numpy")
        tracemalloc.start()
        graph.execute(arrays, backend="numpy")
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        inputs = sum(array.nbytes for array in arrays.values())
        self.assertLessEqual(peak + inputs, graph.memory_plan().peak_memory_bytes)


if __name__ == "__main__":
    unittest.main()
