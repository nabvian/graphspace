from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any
import array
import platform
import sys

from ..core import FLOAT_DTYPES, INT_RANGES, Dims, Node, TensorSpec, _product
from ..failures import ContractViolation, DTypeMismatch, ShapeMismatch
from . import frame_digest


class PythonBackend:
    name = "python"
    version = platform.python_version()

    def session(self):
        return nullcontext()

    def length(self, name: str, data: Any) -> int:
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise DTypeMismatch(
                f"input {name} must be a flat sequence of numbers, got {type(data).__name__}",
                node=name, expected="sequence", actual=type(data).__name__,
            )
        return len(data)

    def coerce(self, name: str, data: Any, spec: TensorSpec, dims: Dims) -> list:
        expected = _product(spec.concrete_shape(dims))
        if self.length(name, data) != expected:
            raise ShapeMismatch(
                f"input {name} has {len(data)} elements, shape {spec.shape} needs {expected}",
                node=name, expected=expected, actual=len(data),
            )
        if spec.dtype in FLOAT_DTYPES:
            if all(type(value) is float for value in data):
                return data if isinstance(data, list) else list(data)
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in data):
                raise DTypeMismatch(
                    f"input {name} must contain only numbers for {spec.dtype}",
                    node=name, expected=spec.dtype, actual=_first_invalid(data, (int, float)),
                )
            return [float(value) for value in data]
        if not all(type(value) is int for value in data) and any(
            isinstance(value, bool) or not isinstance(value, int) for value in data
        ):
            raise DTypeMismatch(
                f"input {name} must contain only ints for {spec.dtype}",
                node=name, expected=spec.dtype, actual=_first_invalid(data, (int,)),
            )
        values = [int(value) for value in data]
        check_range(name, values, spec.dtype)
        return values

    def run(
        self, node: Node, args: list[list], in_shapes: list[tuple[int, ...]], out_shape: tuple[int, ...], out: Any = None,
    ) -> list:
        result = self._apply(node, args, in_shapes)
        check_range(node.output, result, node.output_spec.dtype)
        return result

    def _apply(self, node: Node, args: list[list], in_shapes: list[tuple[int, ...]]) -> list:
        if node.operation == "add":
            return [a + b for a, b in zip(*args)]
        if node.operation == "multiply":
            return [a * b for a, b in zip(*args)]
        if node.operation == "subtract":
            return [a - b for a, b in zip(*args)]
        if node.operation == "relu":
            zero = 0.0 if node.output_spec.dtype in FLOAT_DTYPES else 0
            return [zero if value < zero else value for value in args[0]]
        if node.operation == "reshape":
            return args[0]
        if node.operation == "matmul":
            left, right = args
            (rows, inner), (_, columns) = in_shapes
            return [sum(left[row * inner + k] * right[k * columns + col] for k in range(inner)) for row in range(rows) for col in range(columns)]
        raise ContractViolation(f"{node.name}: unsupported operation {node.operation}", node=node.name, actual=node.operation)

    def digest(self, values: Mapping[str, tuple[str, list]]) -> str:
        entries = []
        for name, (dtype, data) in values.items():
            packed = array.array("d" if dtype in FLOAT_DTYPES else "q", data)
            if sys.byteorder == "big":
                packed.byteswap()
            entries.append((name, dtype, len(data), packed.tobytes()))
        return frame_digest(entries)


def check_range(name: str, values: list, dtype: str) -> None:
    if dtype not in INT_RANGES:
        return
    low, high = INT_RANGES[dtype]
    if values and (min(values) < low or max(values) > high):
        raise DTypeMismatch(
            f"{name}: value outside {dtype} range",
            node=name, expected=[low, high], actual=f"{sum(not low <= value <= high for value in values)} values out of range",
        )


def _first_invalid(data: Sequence, allowed: tuple[type, ...]) -> str:
    for value in data:
        if isinstance(value, bool) or not isinstance(value, allowed):
            return type(value).__name__
    return ""
