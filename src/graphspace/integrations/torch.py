from . import check_dtype, check_shape


def available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def validate_tensor(tensor, spec) -> None:
    check_shape(tuple(tensor.shape), spec)
    check_dtype(str(tensor.dtype).removeprefix("torch."), spec)
