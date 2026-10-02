from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
import hashlib
import json
import platform
import threading

from ._version import __version__
from .claims import Basis, Claim, find_claim
from .failures import ContractViolation, DTypeMismatch, GraphspaceError, ResourceLimitExceeded, ShapeMismatch, UnknownValue


DTYPE_BYTES = {"float32": 4, "float16": 2, "int32": 4, "int64": 8}
LAYOUTS = frozenset({"row_major"})
FLOAT_DTYPES = frozenset({"float32", "float16"})
INT_RANGES = {"int32": (-(2**31), 2**31 - 1), "int64": (-(2**63), 2**63 - 1)}
ELEMENTWISE = frozenset({"add", "multiply", "subtract", "relu"})

Dims = Mapping[str, int]
PREPARED_CACHE_SIZE = 8
BOOKKEEPING_BASE_BYTES = 6912
BOOKKEEPING_INPUT_BYTES = 296
BOOKKEEPING_NODE_BYTES = 368
PYTHON_VERSION = f"{platform.python_implementation()} {platform.python_version()}"


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[Any, ...]
    dtype: str = "float32"
    layout: str = "row_major"
    role: str = "temporary"

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_BYTES:
            raise DTypeMismatch(
                f"unsupported dtype: {self.dtype}",
                expected=sorted(DTYPE_BYTES), actual=self.dtype,
            )
        if self.layout not in LAYOUTS:
            raise ContractViolation(f"unsupported layout: {self.layout}", expected=sorted(LAYOUTS), actual=self.layout)
        if isinstance(self.shape, (str, bytes)) or not isinstance(self.shape, Sequence):
            raise ShapeMismatch(f"invalid shape: {self.shape!r}", expected="sequence of dimensions", actual=self.shape)
        object.__setattr__(self, "shape", tuple(self.shape))
        for value in self.shape:
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ShapeMismatch(
                    f"invalid dimension {value!r} in shape {self.shape}",
                    expected="int or symbolic name", actual=value,
                )
            if isinstance(value, int) and value < 0:
                raise ShapeMismatch(f"negative dimension {value} in shape {self.shape}", expected=">= 0", actual=value)
            if isinstance(value, str) and not value:
                raise ShapeMismatch(f"empty symbolic dimension in shape {self.shape}", expected="non-empty name", actual=value)

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(value for value in self.shape if isinstance(value, str))

    @property
    def nbytes(self) -> int | None:
        return self.nbytes_with()

    def nbytes_with(self, dims: Dims | None = None) -> int | None:
        shape = self.concrete_shape(dims)
        if shape is None:
            return None
        return _product(shape) * DTYPE_BYTES[self.dtype]

    def concrete_shape(self, dims: Dims | None = None) -> tuple[int, ...] | None:
        dims = dims or {}
        if any(isinstance(value, str) and value not in dims for value in self.shape):
            return None
        return tuple(dims[value] if isinstance(value, str) else value for value in self.shape)


@dataclass(frozen=True)
class ResourceContract:
    max_memory_bytes: int | None = None
    deterministic: bool = False

    def __post_init__(self) -> None:
        limit = self.max_memory_bytes
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            raise ContractViolation(
                f"max_memory_bytes must be a non-negative int, got {limit!r}",
                expected="non-negative int", actual=limit,
            )

    @classmethod
    def max_memory(cls, value: int, *, deterministic: bool = False) -> "ResourceContract":
        return cls(value, deterministic)


@dataclass(frozen=True)
class Node:
    name: str
    operation: str
    inputs: tuple[str, ...]
    output: str
    output_spec: TensorSpec


@dataclass(frozen=True)
class MemoryValue:
    name: str
    bytes: int | None
    first_use: int
    last_use: int
    buffer: str = ""


@dataclass(frozen=True)
class MemoryPlan:
    values: tuple[MemoryValue, ...]
    peak_memory_bytes: int | None
    reusable_buffers: int
    bookkeeping_bytes: int = 0


@dataclass(frozen=True)
class Analysis:
    graph_name: str
    memory_plan: MemoryPlan
    claims: tuple[Claim, ...]

    def claim(self, name: str) -> Claim:
        return find_claim(self.claims, name)


@dataclass(frozen=True)
class ExecutionRecord:
    graph_name: str
    backend: str
    runtime: str
    deterministic: bool
    peak_memory_bytes: int | None
    timestamp: str
    graph_sha256: str = ""
    inputs_sha256: str = ""
    output_sha256: str = ""
    python_version: str = ""
    claims: tuple[Claim, ...] = ()

    def claim(self, name: str) -> Claim:
        return find_claim(self.claims, name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Step:
    node: Node
    in_shapes: tuple[tuple[int, ...] | None, ...]
    out_shape: tuple[int, ...] | None
    out_source: str | None
    release: tuple[str, ...]


@dataclass(frozen=True)
class _Prepared:
    plan: MemoryPlan
    steps: tuple[_Step, ...]
    graph_sha256: str
    claims: tuple[Claim, ...]


class Graph:
    def __init__(self, name: str, resources: ResourceContract | None = None) -> None:
        self.name = name
        self.resources = resources or ResourceContract()
        self._inputs: dict[str, TensorSpec] = {}
        self._nodes: list[Node] = []
        self._output: str | None = None
        self._prepared: dict[tuple, _Prepared] = {}
        self._lock = threading.Lock()

    @property
    def resources(self) -> ResourceContract:
        return self._resources

    @resources.setter
    def resources(self, value: ResourceContract) -> None:
        if not isinstance(value, ResourceContract):
            raise ContractViolation(
                f"{self.name}: resources must be a ResourceContract",
                graph=self.name, expected="ResourceContract", actual=type(value).__name__,
            )
        self._resources = value

    @property
    def inputs(self) -> dict[str, TensorSpec]:
        return dict(self._inputs)

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes)

    def input(self, name: str, spec: TensorSpec) -> str:
        if name in self._inputs or any(node.output == name for node in self._nodes):
            raise ContractViolation(
                f"{self.name}: value name already defined: {name}",
                graph=self.name, node=name, remediation="choose a unique input name",
            )
        self._inputs[name] = spec
        return name

    def add(self, left: str, right: str, *, name: str = "add") -> str:
        return self._elementwise("add", left, right, name)

    def multiply(self, left: str, right: str, *, name: str = "multiply") -> str:
        return self._elementwise("multiply", left, right, name)

    def subtract(self, left: str, right: str, *, name: str = "subtract") -> str:
        return self._elementwise("subtract", left, right, name)

    def relu(self, value: str, *, name: str = "relu") -> str:
        source = self._spec(value)
        output = self._unique_output(name)
        self._nodes.append(Node(name, "relu", (value,), output, TensorSpec(source.shape, source.dtype, source.layout)))
        return output

    def reshape(self, value: str, shape: tuple[Any, ...], *, name: str = "reshape") -> str:
        source = self._spec(value)
        target = TensorSpec(shape, source.dtype, source.layout, source.role)
        if _element_count(source.shape) != _element_count(target.shape):
            raise ShapeMismatch(
                f"{name}: cannot prove reshape {source.shape} -> {target.shape} keeps element count",
                graph=self.name, node=name, expected=source.shape, actual=target.shape,
            )
        output = self._unique_output(name)
        self._nodes.append(Node(name, "reshape", (value,), output, target))
        return output

    def matmul(self, left: str, right: str, *, name: str = "matmul") -> str:
        left_spec, right_spec = self._spec(left), self._spec(right)
        if len(left_spec.shape) != 2 or len(right_spec.shape) != 2:
            raise ShapeMismatch(
                f"{name}: matmul requires rank-2 tensors",
                graph=self.name, node=name, expected="rank 2", actual=(left_spec.shape, right_spec.shape),
            )
        if left_spec.shape[1] != right_spec.shape[0]:
            raise ShapeMismatch(
                f"{name}: incompatible inner dimensions",
                graph=self.name, node=name, expected=left_spec.shape[1], actual=right_spec.shape[0],
            )
        if left_spec.dtype != right_spec.dtype:
            raise DTypeMismatch(
                f"{name}: matmul requires matching dtypes",
                graph=self.name, node=name, expected=left_spec.dtype, actual=right_spec.dtype,
            )
        output = self._unique_output(name)
        self._nodes.append(Node(name, "matmul", (left, right), output, TensorSpec((left_spec.shape[0], right_spec.shape[1]), left_spec.dtype)))
        return output

    def output(self, value: str) -> None:
        self._spec(value)
        self._output = value

    def validate(self, dims: Dims | None = None) -> None:
        if self._output is None:
            raise UnknownValue(f"graph {self.name} has no output", graph=self.name, remediation="call graph.output(value)")
        if self.resources.max_memory_bytes is not None:
            self._check_memory(self.memory_plan(dims), dims)

    def _check_memory(self, plan: MemoryPlan, dims: Dims | None) -> None:
        limit = self.resources.max_memory_bytes
        if limit is None:
            return
        if plan.peak_memory_bytes is None:
            unbound = sorted(self._symbols() - set(dims or {}))
            raise ContractViolation(
                f"{self.name}: cannot verify memory limit {limit} with unbound symbolic dimensions {unbound}",
                graph=self.name, expected="bound dimensions", actual=unbound,
                remediation="pass dims to validate() or execute with concrete inputs",
            )
        if plan.peak_memory_bytes > limit:
            raise ResourceLimitExceeded(
                f"{self.name}: peak memory {plan.peak_memory_bytes} exceeds limit {limit}",
                graph=self.name, expected=limit, actual=plan.peak_memory_bytes,
                remediation="raise max_memory_bytes or reduce tensor sizes",
            )

    def analyze(self, dims: Dims | None = None) -> Analysis:
        plan = self.memory_plan(dims)
        output = self._spec(self._output) if self._output is not None else None
        return Analysis(self.name, plan, (
            Claim("shapes_consistent", True, Basis.PROVEN),
            Claim("output_spec", output, Basis.PROVEN),
            Claim("dimensions", dict(dims or {}), Basis.DECLARED),
            Claim("unbound_dimensions", sorted(self._symbols() - set(dims or {})), Basis.PROVEN),
            *self._contract_claims(plan),
        ))

    def memory_plan(self, dims: Dims | None = None) -> MemoryPlan:
        steps = len(self._nodes)
        first_use = {name: 0 for name in self._inputs}
        last_use = {name: steps for name in self._inputs}
        for index, node in enumerate(self._nodes, start=1):
            first_use[node.output] = index
            last_use[node.output] = steps if node.output == self._output else index
            for name in node.inputs:
                last_use[name] = max(last_use[name], index)

        buffer = {name: name for name in self._inputs}
        members: dict[str, list[str]] = {name: [name] for name in self._inputs}
        owned: set[str] = set()
        for index, node in enumerate(self._nodes, start=1):
            target = None
            if node.operation == "reshape":
                target = buffer[node.inputs[0]]
            elif node.operation in ELEMENTWISE:
                for name in node.inputs:
                    candidate = buffer[name]
                    if candidate in owned and max(last_use[member] for member in members[candidate]) == index:
                        target = candidate
                        break
            if target is None:
                target = node.output
                members[target] = []
                owned.add(target)
            buffer[node.output] = target
            members[target].append(node.output)

        reports = tuple(
            MemoryValue(name, self._spec(name).nbytes_with(dims), first_use[name], last_use[name], buffer[name])
            for name in first_use
        )
        sizes = {name: self._spec(name).nbytes_with(dims) for name in members}
        if any(size is None for size in sizes.values()):
            peak = None
        else:
            spans = {
                name: (first_use[name], max(last_use[member] for member in group))
                for name, group in members.items()
            }
            peak = max(
                (sum(sizes[name] for name, (first, last) in spans.items() if first <= step <= last)
                 for step in range(steps + 1)),
                default=0,
            )
        reusable = sum(value.buffer != value.name for value in reports)
        bookkeeping = (
            BOOKKEEPING_BASE_BYTES
            + BOOKKEEPING_INPUT_BYTES * len(self._inputs)
            + BOOKKEEPING_NODE_BYTES * len(self._nodes)
        )
        return MemoryPlan(reports, None if peak is None else peak + bookkeeping, reusable, bookkeeping)

    def execute(
        self, values: Mapping[str, Any], *, backend: str = "python", digests: bool = False,
    ) -> tuple[Any, ExecutionRecord]:
        from .backends import get_backend

        impl = get_backend(backend)
        try:
            return self._execute(impl, values, digests)
        except GraphspaceError as error:
            if error.graph is None:
                error.graph = self.name
            raise

    def _execute(self, impl: Any, values: Mapping[str, Any], digests: bool) -> tuple[Any, ExecutionRecord]:
        missing = sorted(set(self._inputs) - set(values))
        unexpected = sorted(set(values) - set(self._inputs))
        if missing:
            raise UnknownValue(
                f"{self.name}: missing input values: {missing}",
                expected=sorted(self._inputs), actual=sorted(values), remediation=f"provide values for {missing}",
            )
        if unexpected:
            raise UnknownValue(
                f"{self.name}: values for undeclared inputs: {unexpected}",
                expected=sorted(self._inputs), actual=sorted(values), remediation=f"remove {unexpected}",
            )
        dims = self._infer_dims({name: impl.length(name, data) for name, data in values.items()})
        if self._output is None:
            self.validate(dims)
        prepared = self._prepare(dims)
        self._check_memory(prepared.plan, dims)
        computed = {name: impl.coerce(name, values[name], spec, dims) for name, spec in self._inputs.items()}
        digest_claims = []
        if digests:
            digest_claims.append(Claim("inputs_sha256", impl.digest(
                {name: (spec.dtype, computed[name]) for name, spec in self._inputs.items()}
            ), Basis.MEASURED))
        for step in prepared.steps:
            node = step.node
            out = computed[step.out_source] if step.out_source is not None else None
            computed[node.output] = impl.run(node, [computed[name] for name in node.inputs], step.in_shapes, step.out_shape, out)
            for name in step.release:
                del computed[name]
        assert self._output is not None
        result = computed[self._output]
        if digests:
            digest_claims.append(Claim("output_sha256", impl.digest(
                {self._output: (self._spec(self._output).dtype, result)}
            ), Basis.MEASURED))
        return result, ExecutionRecord(
            self.name, impl.name, f"graphspace-{__version__}", self.resources.deterministic,
            prepared.plan.peak_memory_bytes, datetime.now(timezone.utc).isoformat(),
            graph_sha256=prepared.graph_sha256,
            inputs_sha256=digest_claims[0].value if digests else "",
            output_sha256=digest_claims[1].value if digests else "",
            python_version=PYTHON_VERSION,
            claims=(
                Claim("dimensions", dict(dims), Basis.INFERRED),
                *prepared.claims,
                *digest_claims,
                Claim("backend_version", impl.version, Basis.BACKEND_REPORTED),
            ),
        )

    def _prepare(self, dims: Dims) -> _Prepared:
        key = (len(self._inputs), len(self._nodes), self._output, self.resources, tuple(sorted(dims.items())))
        with self._lock:
            if key in self._prepared:
                return self._prepared[key]
        plan = self.memory_plan(dims)
        buffer = {value.name: value.buffer for value in plan.values}
        last_use = {value.name: value.last_use for value in plan.values}
        steps = []
        for index, node in enumerate(self._nodes, start=1):
            out_source = None
            if node.operation != "reshape" and buffer[node.output] != node.output:
                out_source = next(name for name in node.inputs if buffer[name] == buffer[node.output])
            release = tuple(
                name for name in dict.fromkeys((*node.inputs, node.output))
                if last_use[name] == index and name != self._output and name not in self._inputs
            )
            steps.append(_Step(
                node,
                tuple(self._spec(name).concrete_shape(dims) for name in node.inputs),
                node.output_spec.concrete_shape(dims),
                out_source,
                release,
            ))
        dtypes = {spec.dtype for spec in [*self._inputs.values(), *(node.output_spec for node in self._nodes)]}
        claims = (
            Claim("inputs_valid", True, Basis.RUNTIME_CHECKED),
            *([Claim("integer_range", True, Basis.RUNTIME_CHECKED)] if dtypes & set(INT_RANGES) else []),
            *self._contract_claims(plan),
        )
        prepared = _Prepared(plan, tuple(steps), _sha256(self._describe()), claims)
        with self._lock:
            if key not in self._prepared and len(self._prepared) >= PREPARED_CACHE_SIZE:
                del self._prepared[next(iter(self._prepared))]
            self._prepared[key] = prepared
        return prepared

    def _contract_claims(self, plan: MemoryPlan) -> list[Claim]:
        limit = self.resources.max_memory_bytes
        peak = plan.peak_memory_bytes
        within = None if limit is None or peak is None else peak <= limit
        return [
            Claim("peak_memory_bytes", peak, Basis.ESTIMATED),
            Claim("bookkeeping_bytes", plan.bookkeeping_bytes, Basis.ESTIMATED),
            Claim("max_memory_bytes", limit, Basis.DECLARED),
            Claim("within_memory_limit", within, Basis.ESTIMATED),
            Claim("deterministic", self.resources.deterministic, Basis.DECLARED),
        ]

    def _elementwise(self, operation: str, left: str, right: str, name: str) -> str:
        spec_left, spec_right = self._spec(left), self._spec(right)
        if spec_left.shape != spec_right.shape:
            raise ShapeMismatch(
                f"{name}: {operation} requires identical shapes: {spec_left.shape} != {spec_right.shape}",
                graph=self.name, node=name, expected=spec_left.shape, actual=spec_right.shape,
            )
        if spec_left.dtype != spec_right.dtype:
            raise DTypeMismatch(
                f"{name}: {operation} requires identical dtypes: {spec_left.dtype} != {spec_right.dtype}",
                graph=self.name, node=name, expected=spec_left.dtype, actual=spec_right.dtype,
            )
        if spec_left.layout != spec_right.layout:
            raise ShapeMismatch(
                f"{name}: {operation} requires identical layouts: {spec_left.layout} != {spec_right.layout}",
                graph=self.name, node=name, expected=spec_left.layout, actual=spec_right.layout,
            )
        output = self._unique_output(name)
        self._nodes.append(Node(name, operation, (left, right), output, TensorSpec(spec_left.shape, spec_left.dtype, spec_left.layout)))
        return output

    def _infer_dims(self, lengths: Mapping[str, int]) -> dict[str, int]:
        dims: dict[str, int] = {}
        pending = [name for name, spec in self._inputs.items() if spec.symbols]
        while pending:
            progressed = False
            for name in list(pending):
                spec = self._inputs[name]
                unbound = [value for value in spec.shape if isinstance(value, str) and value not in dims]
                if not unbound:
                    pending.remove(name)
                    continue
                if len(unbound) > 1:
                    continue
                known = _product(dims.get(value, value) for value in spec.shape if value != unbound[0])
                length = lengths[name]
                if known == 0 or length % known:
                    raise ShapeMismatch(
                        f"{self.name}: input {name} has {length} elements, incompatible with shape {spec.shape}",
                        node=name, expected=spec.shape, actual=length,
                    )
                dims[unbound[0]] = length // known
                pending.remove(name)
                progressed = True
            if pending and not progressed:
                raise ShapeMismatch(
                    f"{self.name}: cannot infer symbolic dimensions for inputs {sorted(pending)}",
                    expected="one unbound dimension per input", actual=sorted(pending),
                )
        return dims

    def _symbols(self) -> set[str]:
        symbols: set[str] = set()
        for spec in [*self._inputs.values(), *(node.output_spec for node in self._nodes)]:
            symbols |= spec.symbols
        return symbols

    def _describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "resources": [self.resources.max_memory_bytes, self.resources.deterministic],
            "inputs": [[name, list(spec.shape), spec.dtype, spec.layout, spec.role] for name, spec in self._inputs.items()],
            "nodes": [[node.name, node.operation, list(node.inputs), node.output, list(node.output_spec.shape), node.output_spec.dtype] for node in self._nodes],
            "output": self._output,
        }

    def _spec(self, value: str) -> TensorSpec:
        if value in self._inputs:
            return self._inputs[value]
        for node in reversed(self._nodes):
            if node.output == value:
                return node.output_spec
        raise UnknownValue(f"unknown graph value: {value}", graph=self.name, node=value)

    def _unique_output(self, name: str) -> str:
        existing = set(self._inputs) | {node.output for node in self._nodes}
        output = name
        suffix = 1
        while output in existing:
            suffix += 1
            output = f"{name}_{suffix}"
        return output


def _product(values) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _element_count(shape: tuple[Any, ...]) -> tuple[int, tuple[str, ...]]:
    return (
        _product(value for value in shape if isinstance(value, int)),
        tuple(sorted(value for value in shape if isinstance(value, str))),
    )


def _sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

