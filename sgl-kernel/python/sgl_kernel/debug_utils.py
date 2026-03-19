import os
from typing import Any, Callable


def maybe_wrap_sglang_debug(
    func: Callable[..., Any], op_name: str
) -> Callable[..., Any]:
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
    return wrapped
