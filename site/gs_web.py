"""Browser bridge for the Graphspace playground.

Runs inside Pyodide on the GitHub Pages site (and natively in the tests).
Executes the visitor's code with the unmodified ``graphspace`` package, then
reports what the package itself says: nodes, memory plan, analysis claims,
the execution record, and structured failures with the stage they occurred at.
"""
from __future__ import annotations

import contextlib
import io
import json
import traceback

import graphspace
from graphspace import Graph
from graphspace.failures import GraphspaceError


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "_asdict"):
            return _jsonable(value._asdict())
        if hasattr(value, "__dataclass_fields__"):
            return {k: _jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return getattr(value, "value", None) if hasattr(value, "value") and not callable(value.value) else str(value)


def _failure(error: BaseException, stage: str) -> dict:
    if isinstance(error, GraphspaceError):
        return {"stage": stage, "type": type(error).__name__, **_jsonable(error.to_dict())}
    tb = traceback.format_exception_only(type(error), error)
    return {"stage": stage, "type": type(error).__name__, "message": "".join(tb).strip(), "code": None}


def _claims(claims) -> list[dict]:
    return [{"name": c.name, "value": _jsonable(c.value), "basis": getattr(c.basis, "value", str(c.basis))} for c in claims]


def _graph_view(graph: Graph, dims) -> dict:
    view = {
        "name": graph.name,
        "inputs": [{"name": n, "spec": _jsonable(s)} for n, s in graph.inputs.items()],
        "nodes": [{"name": n.name, "operation": n.operation, "inputs": list(n.inputs), "output": n.output,
                   "spec": _jsonable(n.output_spec), "attributes": _jsonable(dict(n.attributes))} for n in graph.nodes],
        "output": getattr(graph, "_output", None),
        "resources": _jsonable(graph.resources),
    }
    try:
        plan = graph.memory_plan(dims)
        view["memory_plan"] = {"values": [_jsonable(v) for v in plan.values], "peak_memory_bytes": plan.peak_memory_bytes,
                               "reusable_buffers": plan.reusable_buffers, "bookkeeping_bytes": plan.bookkeeping_bytes}
    except Exception as error:  # noqa: BLE001 - reported to the visitor
        view["memory_plan_error"] = _failure(error, "memory plan")
    try:
        view["analysis"] = _claims(graph.analyze(dims).claims)
    except Exception as error:  # noqa: BLE001
        view["analysis_error"] = _failure(error, "analysis")
    return view


def run(code: str) -> dict:
    """Execute playground code. It should define ``graph``; ``values`` and ``dims`` are optional."""
    out = io.StringIO()
    namespace: dict = {"__name__": "__playground__"}
    report: dict = {"version": graphspace.__version__, "stages": []}
    with contextlib.redirect_stdout(out):
        try:
            exec(compile(code, "<playground>", "exec"), namespace)
            report["stages"].append({"stage": "construction", "ok": True})
        except Exception as error:  # noqa: BLE001
            report["failure"] = _failure(error, "construction")
            report["stages"].append({"stage": "construction", "ok": False})
        graph = namespace.get("graph") or next((v for v in reversed(list(namespace.values())) if isinstance(v, Graph)), None)
        dims = namespace.get("dims")
        if graph is not None:
            report["graph"] = _graph_view(graph, dims)
        if graph is not None and "failure" not in report:
            try:
                graph.validate(dims)
                report["stages"].append({"stage": "validation", "ok": True})
            except Exception as error:  # noqa: BLE001
                report["failure"] = _failure(error, "validation")
                report["stages"].append({"stage": "validation", "ok": False})
        values = namespace.get("values")
        if graph is not None and values is not None and "failure" not in report:
            try:
                result, record = graph.execute(values, digests=True)
                report["result"] = _jsonable(result)
                report["record"] = _jsonable(record.to_dict())
                report["stages"].append({"stage": "execution", "ok": True})
            except Exception as error:  # noqa: BLE001
                report["failure"] = _failure(error, "execution")
                report["stages"].append({"stage": "execution", "ok": False})
    report["stdout"] = out.getvalue()
    return report
