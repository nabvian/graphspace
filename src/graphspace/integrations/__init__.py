"""Optional integrations for external numerical backends."""

from ..failures import DTypeMismatch, ShapeMismatch


def check_shape(actual: tuple, spec) -> None:
    if len(actual) != len(spec.shape):
        raise ShapeMismatch(f"shape mismatch: {actual} != {spec.shape}", expected=spec.shape, actual=actual)
    bound: dict[str, int] = {}
    for size, expected in zip(actual, spec.shape):
        if isinstance(expected, str):
            if bound.setdefault(expected, size) != size:
                raise ShapeMismatch(
                    f"shape mismatch: {actual} binds {expected} inconsistently for {spec.shape}",
                    expected=spec.shape, actual=actual,
                )
        elif size != expected:
            raise ShapeMismatch(f"shape mismatch: {actual} != {spec.shape}", expected=spec.shape, actual=actual)


def check_dtype(actual: str, spec) -> None:
    if actual != spec.dtype:
        raise DTypeMismatch(f"dtype mismatch: {actual} != {spec.dtype}", expected=spec.dtype, actual=actual)
