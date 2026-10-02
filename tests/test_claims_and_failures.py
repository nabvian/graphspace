import json
import unittest

from graphspace import (
    Basis, BackendUnavailable, ContractViolation, DTypeMismatch, Graph, LowConfidence,
    ResourceContract, ResourceLimitExceeded, ShapeMismatch, TensorSpec, Uncertain, UnknownValue,
)
from graphspace.core import BOOKKEEPING_BASE_BYTES, BOOKKEEPING_INPUT_BYTES, BOOKKEEPING_NODE_BYTES, UFUNC_BUFFER_ELEMENTS


def bookkeeping(inputs, nodes):
    return BOOKKEEPING_BASE_BYTES + BOOKKEEPING_INPUT_BYTES * inputs + BOOKKEEPING_NODE_BYTES * nodes


def buffers(operands, elements, itemsize=4):
    return operands * min(elements, UFUNC_BUFFER_ELEMENTS) * itemsize


def add_graph(spec=TensorSpec((2, 2)), resources=None):
    graph = Graph("add", resources)
    graph.input("a", spec)
    graph.input("b", spec)
    graph.output(graph.add("a", "b"))
    return graph


class TestAnalysis(unittest.TestCase):

    def test_claims_carry_basis(self):
        analysis = add_graph(resources=ResourceContract.max_memory(48 + buffers(3, 4) + bookkeeping(2, 1), deterministic=True)).analyze()
        self.assertEqual(analysis.claim("shapes_consistent").basis, Basis.PROVEN)
        self.assertEqual(analysis.claim("output_spec").value, TensorSpec((2, 2)))
        self.assertEqual(analysis.claim("output_spec").basis, Basis.PROVEN)
        self.assertEqual(analysis.claim("peak_memory_bytes").value, 48 + buffers(3, 4) + bookkeeping(2, 1))
        self.assertEqual(analysis.claim("bookkeeping_bytes").value, bookkeeping(2, 1))
        self.assertEqual(analysis.claim("bookkeeping_bytes").basis, Basis.ESTIMATED)
        self.assertEqual(analysis.claim("peak_memory_bytes").basis, Basis.ESTIMATED)
        self.assertEqual(analysis.claim("max_memory_bytes").basis, Basis.DECLARED)
        self.assertIs(analysis.claim("within_memory_limit").value, True)
        self.assertEqual(analysis.claim("within_memory_limit").basis, Basis.ESTIMATED)
        self.assertEqual(analysis.claim("deterministic").basis, Basis.DECLARED)

    def test_analyze_reports_without_raising(self):
        analysis = add_graph(resources=ResourceContract.max_memory(1)).analyze()
        self.assertIs(analysis.claim("within_memory_limit").value, False)

    def test_unbound_dimensions_reported(self):
        analysis = add_graph(TensorSpec(("N", 2)), ResourceContract.max_memory(100)).analyze()
        self.assertEqual(analysis.claim("unbound_dimensions").value, ["N"])
        self.assertIsNone(analysis.claim("peak_memory_bytes").value)
        self.assertIsNone(analysis.claim("within_memory_limit").value)
        bound = add_graph(TensorSpec(("N", 2))).analyze({"N": 3})
        self.assertEqual(bound.claim("dimensions").basis, Basis.DECLARED)
        self.assertEqual(bound.claim("peak_memory_bytes").value, 72 + buffers(3, 6) + bookkeeping(2, 1))

    def test_unknown_claim_raises_key_error(self):
        with self.assertRaises(KeyError):
            add_graph().analyze().claim("missing")


class TestExecutionClaims(unittest.TestCase):

    def test_record_claims_carry_basis(self):
        _, record = add_graph(TensorSpec(("N", 2))).execute({"a": [1.0] * 4, "b": [2.0] * 4}, digests=True)
        self.assertEqual(record.claim("dimensions").value, {"N": 2})
        self.assertEqual(record.claim("dimensions").basis, Basis.INFERRED)
        self.assertEqual(record.claim("inputs_valid").basis, Basis.RUNTIME_CHECKED)
        self.assertEqual(record.claim("peak_memory_bytes").basis, Basis.ESTIMATED)
        self.assertEqual(record.claim("deterministic").basis, Basis.DECLARED)
        self.assertEqual(record.claim("inputs_sha256").value, record.inputs_sha256)
        self.assertEqual(record.claim("output_sha256").basis, Basis.MEASURED)
        self.assertEqual(record.claim("backend_version").basis, Basis.BACKEND_REPORTED)

    def test_integer_range_claim_only_for_int_graphs(self):
        _, float_record = add_graph().execute({"a": [1.0] * 4, "b": [2.0] * 4})
        with self.assertRaises(KeyError):
            float_record.claim("integer_range")
        _, int_record = add_graph(TensorSpec((1,), "int32")).execute({"a": [1], "b": [2]})
        self.assertEqual(int_record.claim("integer_range").basis, Basis.RUNTIME_CHECKED)

    def test_record_serializes_to_json(self):
        _, record = add_graph().execute({"a": [1.0] * 4, "b": [2.0] * 4})
        payload = json.loads(json.dumps(record.to_dict()))
        self.assertEqual(payload["graph_name"], "add")
        basis = {claim["name"]: claim["basis"] for claim in payload["claims"]}
        self.assertEqual(basis["peak_memory_bytes"], "estimated")


class TestStructuredFailures(unittest.TestCase):

    def test_construction_failure_fields(self):
        graph = Graph("g")
        graph.input("a", TensorSpec((2, 2)))
        graph.input("b", TensorSpec((2, 3)))
        with self.assertRaises(ShapeMismatch) as context:
            graph.add("a", "b", name="sum")
        error = context.exception
        self.assertEqual(error.code, "shape_mismatch")
        self.assertEqual(error.graph, "g")
        self.assertEqual(error.node, "sum")
        self.assertEqual(error.expected, (2, 2))
        self.assertEqual(error.actual, (2, 3))

    def test_execution_failure_names_graph_and_input(self):
        with self.assertRaises(ShapeMismatch) as context:
            add_graph().execute({"a": [1.0] * 3, "b": [1.0] * 4})
        error = context.exception
        self.assertEqual(error.graph, "add")
        self.assertEqual(error.node, "a")
        self.assertEqual((error.expected, error.actual), (4, 3))

    def test_dtype_failure_reports_offending_type(self):
        with self.assertRaises(DTypeMismatch) as context:
            add_graph().execute({"a": [1.0, "x", 1.0, 1.0], "b": [1.0] * 4})
        self.assertEqual(context.exception.actual, "str")

    def test_resource_failure_has_remediation(self):
        graph = add_graph(resources=ResourceContract.max_memory(10))
        with self.assertRaises(ResourceLimitExceeded) as context:
            graph.validate()
        error = context.exception
        self.assertEqual((error.expected, error.actual), (10, 48 + buffers(3, 4) + bookkeeping(2, 1)))
        self.assertTrue(error.remediation)

    def test_missing_input_remediation(self):
        with self.assertRaises(UnknownValue) as context:
            add_graph().execute({"a": [1.0] * 4})
        self.assertIn("b", context.exception.remediation)

    def test_unknown_backend_lists_options(self):
        with self.assertRaises(BackendUnavailable) as context:
            add_graph().execute({"a": [1.0] * 4, "b": [1.0] * 4}, backend="cuda")
        self.assertEqual(context.exception.expected, ["python", "numpy"])
        self.assertEqual(context.exception.actual, "cuda")

    def test_low_confidence(self):
        with self.assertRaises(LowConfidence) as context:
            Uncertain("cat", 0.5).require_confidence(0.9)
        error = context.exception
        self.assertIsInstance(error, ContractViolation)
        self.assertEqual((error.expected, error.actual), (0.9, 0.5))
        self.assertEqual(error.code, "low_confidence")

    def test_to_dict_is_json_serializable(self):
        with self.assertRaises(ShapeMismatch) as context:
            add_graph().execute({"a": [1.0] * 3, "b": [1.0] * 4})
        payload = json.loads(json.dumps(context.exception.to_dict()))
        self.assertEqual(payload["code"], "shape_mismatch")
        self.assertEqual(payload["graph"], "add")


if __name__ == "__main__":
    unittest.main()
