from . import check_dtype, check_shape


def available() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def validate_array(array, spec) -> None:
    check_shape(tuple(array.shape), spec)
    check_dtype(str(array.dtype), spec)
