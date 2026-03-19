import os
from typing import Any, Callable, TypeVar, cast, overload

F = TypeVar("F", bound=Callable[..., Any])


def _wrap_sglang_debug(func: F, op_name: str) -> F:
    try:
        if int(os.environ.get("SGLANG_API_LOGLEVEL", "0")) == 0:
            return func
    except Exception:
        return func

    try:
        from sglang.api_logging import sglang_debug_api
    except Exception:
        return func

    if getattr(func, "_sglang_debug_wrapped", False):
        return func

    wrapped = sglang_debug_api(func, op_name=op_name)
    setattr(wrapped, "_sglang_debug_wrapped", True)
    return cast(F, wrapped)


@overload
def maybe_wrap_sglang_debug(func: F, op_name: str) -> F: ...


@overload
def maybe_wrap_sglang_debug(*, op_name: str) -> Callable[[F], F]: ...


def maybe_wrap_sglang_debug(
    func: F | None = None, op_name: str | None = None
) -> F | Callable[[F], F]:
    if op_name is None:
        raise TypeError("op_name must be provided")

    if func is None:
        return lambda wrapped_func: _wrap_sglang_debug(wrapped_func, op_name)

    return _wrap_sglang_debug(func, op_name)
