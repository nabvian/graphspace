from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any
import array
import math
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
        result = self._apply(node, args, in_shapes, out_shape)
        check_range(node.output, result, node.output_spec.dtype)
        return result

    def _apply(self, node: Node, args: list[list], in_shapes: list[tuple[int, ...]], out_shape: tuple[int, ...]) -> list:
        operation = node.operation
        if operation in BINARY:
            left, right = args
            if in_shapes[0] != out_shape:
                left = [left[index] for index in broadcast_indices(in_shapes[0], out_shape)]
            if in_shapes[1] != out_shape:
                right = [right[index] for index in broadcast_indices(in_shapes[1], out_shape)]
            function = BINARY[operation]
            return [function(a, b) for a, b in zip(left, right)]
        if operation == "relu":
            zero = 0.0 if node.output_spec.dtype in FLOAT_DTYPES else 0
            return [zero if value < zero else value for value in args[0]]
        if operation == "scale":
            factor = node.attribute("factor")
            return [value * factor for value in args[0]]
        if operation == "reshape":
            return args[0]
        if operation == "transpose":
            return [args[0][index] for index in transpose_indices(in_shapes[0], node.attribute("axes"))]
        if operation == "matmul":
            left, right = args
            (rows, inner), (_, columns) = in_shapes
            return [sum(left[row * inner + k] * right[k * columns + col] for k in range(inner)) for row in range(rows) for col in range(columns)]
        if operation == "softmax":
            return [value for row in _rows(args[0], out_shape) for value in _softmax(row)]
        if operation == "layer_norm":
            values, gamma, beta = args
            eps = node.attribute("eps")
            return [value for row in _rows(values, out_shape) for value in _layer_norm(row, gamma, beta, eps)]
        raise ContractViolation(f"{node.name}: unsupported operation {operation}", node=node.name, actual=operation)

    def digest(self, values: Mapping[str, tuple[str, list]]) -> str:
        entries = []
        for name, (dtype, data) in values.items():
            packed = array.array("d" if dtype in FLOAT_DTYPES else "q", data)
            if sys.byteorder == "big":
                packed.byteswap()
            entries.append((name, dtype, len(data), packed.tobytes()))
        return frame_digest(entries)


def _divide(a: float, b: float) -> float:
    if b != 0:
        return a / b
    if a != a or a == 0:
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


BINARY = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": _divide,
}


def _strides(shape: tuple[int, ...]) -> list[int]:
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return strides


def _expand(sizes: tuple[int, ...], strides: list[int]) -> list[int]:
    indices = [0]
    for size, stride in zip(sizes, strides):
        indices = [base + step * stride for base in indices for step in range(size)]
    return indices


def broadcast_indices(in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> list[int]:
    padded = (1,) * (len(out_shape) - len(in_shape)) + tuple(in_shape)
    strides = [0 if size == 1 else stride for size, stride in zip(padded, _strides(padded))]
    return _expand(out_shape, strides)


def transpose_indices(in_shape: tuple[int, ...], axes: tuple[int, ...]) -> list[int]:
    strides = _strides(in_shape)
    return _expand(tuple(in_shape[axis] for axis in axes), [strides[axis] for axis in axes])


def _rows(values: list, shape: tuple[int, ...]):
    width = shape[-1]
    for start in range(0, len(values), width or 1):
        yield values[start:start + width]


def _softmax(row: list) -> list:
    if any(value != value for value in row):
        return [math.nan] * len(row)
    peak = max(row, default=0.0)
    exponents = [math.exp(value - peak) if value - peak == value - peak else math.nan for value in row]
    total = sum(exponents)
    return [_divide(value, total) for value in exponents]


def _layer_norm(row: list, gamma: list, beta: list, eps: float) -> list:
    width = len(row)
    mean = sum(row) / width if width else math.nan
    centered = [value - mean for value in row]
    deviation = math.sqrt(sum(value * value for value in centered) / width + eps) if width else math.nan
    return [value / deviation * g + b for value, g, b in zip(centered, gamma, beta)]


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
