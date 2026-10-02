import unittest
from types import SimpleNamespace

from graphspace import Graph, ResourceContract, TensorSpec, Uncertain
from graphspace.failures import (
    ContractViolation, DTypeMismatch, ResourceLimitExceeded, ShapeMismatch, UnknownValue,
)
from graphspace.core import BOOKKEEPING_BASE_BYTES, BOOKKEEPING_INPUT_BYTES, BOOKKEEPING_NODE_BYTES
from graphspace.integrations import numpy as numpy_adapter, torch as torch_adapter


def bookkeeping(inputs, nodes):
    return BOOKKEEPING_BASE_BYTES + BOOKKEEPING_INPUT_BYTES * inputs + BOOKKEEPING_NODE_BYTES * nodes


def add_graph(spec=TensorSpec((2, 2)), resources=None):
    graph = Graph("add", resources)
    graph.input("a", spec)
    graph.input("b", spec)
    graph.output(graph.add("a", "b"))
    return graph


class TestGraphspace(unittest.TestCase):

    def test_graph_executes_and_records_memory(self):
        result, record = add_graph().execute({"a": [1.0] * 4, "b": [2.0] * 4})
        self.assertEqual(result, [3.0] * 4)
        self.assertEqual(record.peak_memory_bytes, 48 + bookkeeping(2, 1))

    def test_shape_validation(self):
        graph = Graph("bad")
        graph.input("a", TensorSpec((2, 2)))
        graph.input("b", TensorSpec((2, 3)))
        with self.assertRaises(ShapeMismatch):
            graph.add("a", "b")

    def test_resource_contract(self):
        graph = Graph("limited", ResourceContract.max_memory(1))
        graph.input("a", TensorSpec((2, 2)))
        graph.output("a")
        with self.assertRaises(ResourceLimitExceeded):
            graph.validate()

    def test_uncertainty_threshold(self):
        self.assertEqual(Uncertain("ok", 0.95).require_confidence(0.9), "ok")

    def test_matmul_executes(self):
        graph = Graph("matmul")
        graph.input("a", TensorSpec((2, 2)))
        graph.input("b", TensorSpec((2, 2)))
        graph.output(graph.matmul("a", "b"))
        result, _ = graph.execute({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        self.assertEqual(result, [19, 22, 43, 50])

    def test_reshape_preserves_elements(self):
        graph = Graph("reshape")
        graph.input("x", TensorSpec((2, 2)))
        graph.output(graph.reshape("x", (4,)))
        result, _ = graph.execute({"x": [1, 2, 3, 4]})
        self.assertEqual(result, [1, 2, 3, 4])

    def test_arithmetic_operations(self):
        graph = Graph("arithmetic")
        spec = TensorSpec((1, 2))
        graph.input("a", spec)
        graph.input("b", spec)
        graph.output(graph.subtract(graph.multiply("a", "b"), "b"))
        result, _ = graph.execute({"a": [3, 4], "b": [2, 5]})
        self.assertEqual(result, [4, 15])


class TestInputValidation(unittest.TestCase):

    def test_wrong_length_input_is_rejected(self):
        with self.assertRaises(ShapeMismatch):
            add_graph().execute({"a": [1, 2], "b": [1, 2, 3, 4, 5, 6]})

    def test_missing_input_is_structured_error(self):
        with self.assertRaises(UnknownValue):
            add_graph().execute({"a": [1] * 4})

    def test_undeclared_input_is_rejected(self):
        with self.assertRaises(UnknownValue):
            add_graph().execute({"a": [1] * 4, "b": [1] * 4, "c": [1] * 4})

    def test_non_numeric_values_are_rejected(self):
        with self.assertRaises(DTypeMismatch):
            add_graph().execute({"a": ["x"] * 4, "b": ["y"] * 4})
        with self.assertRaises(DTypeMismatch):
            add_graph().execute({"a": [True] * 4, "b": [1] * 4})
        with self.assertRaises(DTypeMismatch):
            add_graph().execute({"a": "abcd", "b": [1] * 4})

    def test_float_dtype_outputs_floats(self):
        result, _ = add_graph().execute({"a": [1, 2, 3, 4], "b": [1, 1, 1, 1]})
        self.assertTrue(all(isinstance(value, float) for value in result))

    def test_int_dtype_keeps_ints(self):
        graph = Graph("ints")
        graph.input("x", TensorSpec((1, 2), "int32"))
        graph.output(graph.relu("x"))
        result, _ = graph.execute({"x": [-1, 2]})
        self.assertEqual(result, [0, 2])
        self.assertTrue(all(type(value) is int for value in result))

    def test_int_dtype_rejects_floats_and_overflow(self):
        graph = add_graph(TensorSpec((1,), "int32"))
        with self.assertRaises(DTypeMismatch):
            graph.execute({"a": [1.5], "b": [1]})
        with self.assertRaises(DTypeMismatch):
            graph.execute({"a": [2**31], "b": [0]})
        with self.assertRaises(DTypeMismatch):
            graph.execute({"a": [2**31 - 1], "b": [1]})

    def test_range_errors_do_not_expose_values(self):
        graph = add_graph(TensorSpec((3,), "int32"))
        with self.assertRaises(DTypeMismatch) as context:
            graph.execute({"a": [5, 2**40, 2**41], "b": [0, 0, 0]})
        self.assertEqual(context.exception.actual, "2 values out of range")
        self.assertNotIn(str(2**40), str(context.exception.to_dict()))

    def test_input_cannot_be_redefined(self):
        graph = Graph("dup")
        graph.input("a", TensorSpec((2, 2)))
        with self.assertRaises(ContractViolation):
            graph.input("a", TensorSpec((9, 9)))
        graph.output(graph.relu("a"))
        with self.assertRaises(ContractViolation):
            graph.input("relu", TensorSpec((1,)))


class TestTensorSpec(unittest.TestCase):

    def test_negative_bool_and_empty_dimensions_rejected(self):
        for shape in [(-3, 4), (True, 4), ("", 4), (1.5,), "ab"]:
            with self.subTest(shape=shape), self.assertRaises(ShapeMismatch):
                TensorSpec(shape)

    def test_zero_dimension_allowed(self):
        self.assertEqual(TensorSpec((0, 4)).nbytes, 0)

    def test_list_shape_normalized_to_tuple(self):
        self.assertEqual(TensorSpec([2, 3]).shape, (2, 3))

    def test_role_difference_does_not_block_elementwise(self):
        graph = Graph("roles")
        graph.input("a", TensorSpec((2, 2), role="activation"))
        graph.input("b", TensorSpec((2, 2)))
        graph.output(graph.add("a", "b"))
        result, _ = graph.execute({"a": [1] * 4, "b": [1] * 4})
        self.assertEqual(result, [2.0] * 4)

    def test_dtype_difference_is_dtype_error(self):
        graph = Graph("dtypes")
        graph.input("a", TensorSpec((2,), "float32"))
        graph.input("b", TensorSpec((2,), "int32"))
        with self.assertRaises(DTypeMismatch):
            graph.add("a", "b")

    def test_negative_memory_limit_rejected(self):
        with self.assertRaises(ContractViolation):
            ResourceContract.max_memory(-1)


class TestSymbolicShapes(unittest.TestCase):

    def test_symbolic_shape_cannot_bypass_memory_limit(self):
        graph = add_graph(TensorSpec(("N", 1_000_000)), ResourceContract.max_memory(1))
        with self.assertRaises(ContractViolation):
            graph.validate()
        with self.assertRaises(ResourceLimitExceeded):
            graph.validate({"N": 1})

    def test_symbolic_dims_bound_at_execute(self):
        graph = add_graph(TensorSpec(("N", 2)), ResourceContract.max_memory(72 + bookkeeping(2, 1)))
        result, record = graph.execute({"a": [1] * 6, "b": [2] * 6})
        self.assertEqual(result, [3.0] * 6)
        self.assertEqual(record.peak_memory_bytes, 3 * 6 * 4 + bookkeeping(2, 1))

    def test_symbolic_execute_enforces_memory_limit(self):
        graph = add_graph(TensorSpec(("N", 2)), ResourceContract.max_memory(71 + bookkeeping(2, 1)))
        with self.assertRaises(ResourceLimitExceeded):
            graph.execute({"a": [1] * 6, "b": [2] * 6})

    def test_symbolic_length_must_divide(self):
        with self.assertRaises(ShapeMismatch):
            add_graph(TensorSpec(("N", 4))).execute({"a": [1] * 6, "b": [1] * 6})

    def test_inconsistent_symbol_binding_rejected(self):
        graph = Graph("bind")
        graph.input("a", TensorSpec(("N",)))
        graph.input("b", TensorSpec(("N",)))
        graph.output(graph.add("a", "b"))
        with self.assertRaises(ShapeMismatch):
            graph.execute({"a": [1, 2], "b": [1, 2, 3]})

    def test_symbolic_reshape_must_preserve_count(self):
        graph = Graph("reshape")
        graph.input("x", TensorSpec(("N", 4)))
        with self.assertRaises(ShapeMismatch):
            graph.reshape("x", (3,))
        graph.output(graph.reshape("x", ("N", 2, 2)))
        result, _ = graph.execute({"x": list(range(8))})
        self.assertEqual(result, [float(value) for value in range(8)])

    def test_symbolic_matmul_executes(self):
        graph = Graph("matmul")
        graph.input("a", TensorSpec(("N", "K")))
        graph.input("b", TensorSpec(("K", 2)))
        graph.output(graph.matmul("a", "b"))
        result, _ = graph.execute({"a": [1, 2, 3, 4, 5, 6], "b": [1, 0, 0, 1, 1, 1]})
        self.assertEqual(result, [4.0, 5.0, 10.0, 11.0])


class TestMemoryPlan(unittest.TestCase):

    def test_reusable_buffers_from_liveness(self):
        graph = Graph("chain")
        graph.input("x", TensorSpec((4,)))
        first = graph.relu("x")
        second = graph.relu(first)
        graph.output(graph.relu(second))
        plan = graph.memory_plan()
        self.assertEqual(plan.peak_memory_bytes - plan.bookkeeping_bytes, 32)
        self.assertEqual(plan.bookkeeping_bytes, bookkeeping(1, 3))
        self.assertEqual(plan.reusable_buffers, 2)
        buffers = {value.name: value.buffer for value in plan.values}
        self.assertEqual(buffers, {"x": "x", "relu": "relu", "relu_2": "relu", "relu_3": "relu"})

    def test_caller_inputs_are_never_reused(self):
        graph = Graph("inputs")
        graph.input("x", TensorSpec((4,)))
        graph.output(graph.relu(graph.reshape("x", (2, 2))))
        buffers = {value.name: value.buffer for value in graph.memory_plan().values}
        self.assertEqual(buffers, {"x": "x", "reshape": "x", "relu": "relu"})

    def test_matmul_gets_its_own_buffer(self):
        graph = Graph("mm")
        graph.input("a", TensorSpec((2, 2)))
        graph.input("b", TensorSpec((2, 2)))
        hidden = graph.relu(graph.add("a", "b"))
        graph.output(graph.matmul(hidden, "b"))
        buffers = {value.name: value.buffer for value in graph.memory_plan().values}
        self.assertEqual(buffers["relu"], "add")
        self.assertEqual(buffers["matmul"], "matmul")

    def test_pipeline_peak_counts_shared_buffers_once(self):
        graph = Graph("pipeline")
        graph.input("x", TensorSpec((4,)))
        graph.input("y", TensorSpec((4,)))
        value = "x"
        for step in range(6):
            value = graph.add(value, "y") if step % 2 == 0 else graph.multiply(value, "y")
        graph.output(value)
        plan = graph.memory_plan()
        self.assertEqual(plan.peak_memory_bytes - plan.bookkeeping_bytes, 3 * 16)
        self.assertEqual(plan.reusable_buffers, 5)


class TestPreparedCache(unittest.TestCase):

    def test_graph_changes_after_execution_are_used(self):
        graph = add_graph()
        first, first_record = graph.execute({"a": [1.0] * 4, "b": [2.0] * 4})
        graph.output(graph.multiply("add", "b"))
        second, second_record = graph.execute({"a": [1.0] * 4, "b": [2.0] * 4})
        self.assertEqual(first, [3.0] * 4)
        self.assertEqual(second, [6.0] * 4)
        self.assertNotEqual(first_record.graph_sha256, second_record.graph_sha256)

    def test_output_change_alone_is_used(self):
        graph = add_graph()
        graph.execute({"a": [1.0] * 4, "b": [2.0] * 4})
        graph.output("a")
        result, _ = graph.execute({"a": [1.0] * 4, "b": [2.0] * 4})
        self.assertEqual(result, [1.0] * 4)

    def test_resource_change_after_execution_is_enforced(self):
        graph = add_graph()
        graph.execute({"a": [1.0] * 4, "b": [2.0] * 4})
        graph.resources = ResourceContract.max_memory(1)
        with self.assertRaises(ResourceLimitExceeded):
            graph.execute({"a": [1.0] * 4, "b": [2.0] * 4})

    def test_symbolic_dims_cached_per_binding(self):
        graph = add_graph(TensorSpec(("N", 2)))
        _, small = graph.execute({"a": [1.0] * 2, "b": [1.0] * 2})
        _, large = graph.execute({"a": [1.0] * 8, "b": [1.0] * 8})
        _, small_again = graph.execute({"a": [1.0] * 2, "b": [1.0] * 2})
        self.assertEqual(small.peak_memory_bytes, small_again.peak_memory_bytes)
        self.assertEqual(large.peak_memory_bytes - small.peak_memory_bytes, 3 * 6 * 4)


class TestProvenance(unittest.TestCase):

    def test_record_hashes_graph_inputs_and_output(self):
        _, first = add_graph().execute({"a": [1] * 4, "b": [2] * 4}, digests=True)
        _, again = add_graph().execute({"a": [1] * 4, "b": [2] * 4}, digests=True)
        _, other = add_graph().execute({"a": [1] * 4, "b": [3] * 4}, digests=True)
        for record in (first, again, other):
            self.assertEqual(len(record.graph_sha256), 64)
        self.assertEqual(first.graph_sha256, again.graph_sha256)
        self.assertEqual(first.inputs_sha256, again.inputs_sha256)
        self.assertEqual(first.output_sha256, again.output_sha256)
        self.assertNotEqual(first.inputs_sha256, other.inputs_sha256)
        self.assertNotEqual(first.output_sha256, other.output_sha256)
        self.assertTrue(first.python_version)

    def test_content_digests_are_opt_in(self):
        _, record = add_graph().execute({"a": [1] * 4, "b": [2] * 4})
        self.assertEqual(len(record.graph_sha256), 64)
        self.assertEqual((record.inputs_sha256, record.output_sha256), ("", ""))
        with self.assertRaises(KeyError):
            record.claim("inputs_sha256")


class TestUncertainty(unittest.TestCase):

    def test_invalid_confidence_rejected(self):
        for confidence in [1.5, -0.1, float("nan"), "0.9", True]:
            with self.subTest(confidence=confidence), self.assertRaises(ContractViolation):
                Uncertain("x", confidence)

    def test_require_calibrated(self):
        with self.assertRaises(ContractViolation):
            Uncertain("x", 0.99).require_confidence(0.9, require_calibrated=True)
        self.assertEqual(Uncertain("x", 0.99, calibrated=True).require_confidence(0.9, require_calibrated=True), "x")

    def test_below_threshold_rejected(self):
        with self.assertRaises(ContractViolation):
            Uncertain("x", 0.5).require_confidence(0.9)


class TestAdapters(unittest.TestCase):

    def test_torch_adapter_checks_dtype(self):
        spec = TensorSpec((2, 2), "float32")
        torch_adapter.validate_tensor(SimpleNamespace(shape=(2, 2), dtype="torch.float32"), spec)
        with self.assertRaises(DTypeMismatch):
            torch_adapter.validate_tensor(SimpleNamespace(shape=(2, 2), dtype="torch.float16"), spec)

    def test_adapters_check_shape_with_symbols(self):
        spec = TensorSpec(("N", "N"), "float32")
        numpy_adapter.validate_array(SimpleNamespace(shape=(3, 3), dtype="float32"), spec)
        with self.assertRaises(ShapeMismatch):
            numpy_adapter.validate_array(SimpleNamespace(shape=(3, 4), dtype="float32"), spec)
        with self.assertRaises(ShapeMismatch):
            numpy_adapter.validate_array(SimpleNamespace(shape=(3,), dtype="float32"), spec)


if __name__ == "__main__":
    unittest.main()
