from typing import Any


class GraphspaceError(Exception):
    """Base class for structured Graphspace failures."""

    code = "graphspace_error"

    def __init__(
        self,
        message: str,
        *,
        graph: str | None = None,
        node: str | None = None,
        expected: Any = None,
        actual: Any = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.graph = graph
        self.node = node
        self.expected = expected
        self.actual = actual
        self.remediation = remediation

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "graph": self.graph,
            "node": self.node,
            "expected": self.expected,
            "actual": self.actual,
            "remediation": self.remediation,
        }


class ShapeMismatch(GraphspaceError):
    code = "shape_mismatch"


class DTypeMismatch(GraphspaceError):
    code = "dtype_mismatch"


class ResourceLimitExceeded(GraphspaceError):
    code = "resource_limit_exceeded"


class UnknownValue(GraphspaceError):
    code = "unknown_value"


class ContractViolation(GraphspaceError):
    code = "contract_violation"


class LowConfidence(ContractViolation):
    code = "low_confidence"


class BackendUnavailable(GraphspaceError):
    code = "backend_unavailable"
