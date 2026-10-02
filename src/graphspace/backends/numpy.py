from collections.abc import Mapping
from typing import Any

from ..core import FLOAT_DTYPES, INT_RANGES, Dims, Node, TensorSpec, _product
from ..failures import BackendUnavailable, ContractViolation, DTypeMismatch, ShapeMismatch
from . import frame_digest
from .python import PythonBackend


class NumpyBackend:
    name = "numpy"

    def __init__(self) -> None:
        try:
            import numpy
        except ImportError as error:
            raise BackendUnavailable(
                "backend 'numpy' requires NumPy",
                actual="numpy not installed", remediation="pip install 'graphspace[numpy]'",
            ) from error
        self.np = numpy
        self.version = numpy.__version__
        self._python = PythonBackend()

    def session(self):
        return self.np.errstate(all="ignore")

    def length(self, name: str, data: Any) -> int:
        if isinstance(data, self.np.ndarray):
            return int(data.size)
        return self._python.length(name, data)

    def coerce(self, name: str, data: Any, spec: TensorSpec, dims: Dims):
        np = self.np
        shape = spec.concrete_shape(dims)
        if not isinstance(data, np.ndarray):
            return np.array(self._python.coerce(name, data, spec, dims), dtype=spec.dtype).reshape(shape)
        if isinstance(data, np.ma.MaskedArray):
            raise DTypeMismatch(
                f"input {name} is a masked array",
                node=name, expected="ndarray", actual="MaskedArray", remediation="pass data.filled(value) or data.compressed()",
            )
        data = np.asarray(data)
        if data.size != _product(shape):
            raise ShapeMismatch(
                f"input {name} has {data.size} elements, shape {spec.shape} needs {_product(shape)}",
                node=name, expected=_product(shape), actual=int(data.size),
            )
        if data.ndim > 1 and data.shape != shape:
            raise ShapeMismatch(f"input {name} has shape {data.shape}, expected {shape}", node=name, expected=shape, actual=data.shape)
        allowed = "fiu" if spec.dtype in FLOAT_DTYPES else "iu"
        if data.dtype.kind not in allowed:
            raise DTypeMismatch(
                f"input {name} has dtype {data.dtype}, not usable as {spec.dtype}",
                node=name, expected=spec.dtype, actual=str(data.dtype),
            )
        if spec.dtype in INT_RANGES and data.size:
            low, high = INT_RANGES[spec.dtype]
            if int(data.min()) < low or int(data.max()) > high:
                raise DTypeMismatch(
                    f"input {name}: value outside {spec.dtype} range",
                    node=name, expected=[low, high], actual=f"{int(((data < low) | (data > high)).sum())} values out of range",
                )
        array = data if data.dtype == spec.dtype else data.astype(spec.dtype)
        return array.reshape(shape)

    def run(self, node: Node, args: list, in_shapes: list[tuple[int, ...]], out_shape: tuple[int, ...], out: Any = None):
        np = self.np
        dtype = node.output_spec.dtype
        if dtype in INT_RANGES and node.operation in {"add", "multiply", "subtract", "matmul"}:
            wide = np.int64 if dtype == "int32" and node.operation != "matmul" else object
            result = self._apply(node, [arg.astype(wide) for arg in args], out_shape, None)
            low, high = INT_RANGES[dtype]
            if result.size and (result.min() < low or result.max() > high):
                raise DTypeMismatch(
                    f"{node.output}: value outside {dtype} range",
                    node=node.output, expected=[low, high], actual=f"{int(((result < low) | (result > high)).sum())} values out of range",
                )
            return result.astype(dtype)
        return self._apply(node, args, out_shape, out)

    def _apply(self, node: Node, args: list, out_shape: tuple[int, ...], out: Any):
        np = self.np
        if node.operation == "add":
            return np.add(*args, out=out, order="C")
        if node.operation == "multiply":
            return np.multiply(*args, out=out, order="C")
        if node.operation == "subtract":
            return np.subtract(*args, out=out, order="C")
        if node.operation == "divide":
            return np.divide(*args, out=out, order="C")
        if node.operation == "relu":
            return np.maximum(args[0], args[0].dtype.type(0), out=out, order="C")
        if node.operation == "scale":
            return np.multiply(args[0], args[0].dtype.type(node.attribute("factor")), out=out, order="C")
        if node.operation == "reshape":
            return args[0].reshape(out_shape)
        if node.operation == "transpose":
            return args[0].transpose(node.attribute("axes"))
        if node.operation == "matmul":
            return np.matmul(*args, out=out)
        if node.operation == "softmax":
            values = args[0]
            result = np.subtract(values, values.max(axis=-1, keepdims=True, initial=-np.inf), out=out, order="C")
            np.exp(result, out=result)
            return np.divide(result, result.sum(axis=-1, keepdims=True), out=result)
        if node.operation == "layer_norm":
            values, gamma, beta = args
            width = values.shape[-1]
            result = np.subtract(values, values.mean(axis=-1, keepdims=True), out=out, order="C")
            rows = result.reshape(-1, width)
            variance = np.einsum("ij,ij->i", rows, rows).reshape(result.shape[:-1] + (1,))
            np.divide(variance, width, out=variance)
            np.add(variance, node.attribute("eps"), out=variance)
            np.sqrt(variance, out=variance)
            np.divide(result, variance, out=result)
            np.multiply(result, gamma, out=result)
            return np.add(result, beta, out=result)
        raise ContractViolation(f"{node.name}: unsupported operation {node.operation}", node=node.name, actual=node.operation)

    def digest(self, values: Mapping[str, tuple[str, Any]]) -> str:
        np = self.np
        entries = []
        for name, (dtype, data) in values.items():
            wide = np.ascontiguousarray(data, dtype="<f8" if dtype in FLOAT_DTYPES else "<i8")
            entries.append((name, dtype, int(wide.size), wide.tobytes()))
        return frame_digest(entries)
