from collections.abc import Iterable
import hashlib
import json

from ..failures import BackendUnavailable

BACKENDS = ("python", "numpy")
_INSTANCES: dict = {}


def get_backend(name: str):
    if name in _INSTANCES:
        return _INSTANCES[name]
    if name == "python":
        from .python import PythonBackend
        return _INSTANCES.setdefault(name, PythonBackend())
    if name == "numpy":
        from .numpy import NumpyBackend
        return _INSTANCES.setdefault(name, NumpyBackend())
    raise BackendUnavailable(f"unknown backend {name!r}", expected=list(BACKENDS), actual=name)


def frame_digest(entries: Iterable[tuple[str, str, int, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, dtype, length, payload in sorted(entries, key=lambda entry: entry[0]):
        header = json.dumps([name, dtype, length]).encode()
        digest.update(len(header).to_bytes(8, "little") + header + payload)
    return digest.hexdigest()
