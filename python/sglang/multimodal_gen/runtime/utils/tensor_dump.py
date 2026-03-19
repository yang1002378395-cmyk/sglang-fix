from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

_DUMP_DIR_ENV = "SGLANG_DUMP_TENSORS_DIR"
_DUMP_COUNTS: dict[tuple[str, str], int] = defaultdict(int)
_METADATA_WRITTEN: set[str] = set()


def is_tensor_dump_enabled() -> bool:
    return bool(os.environ.get(_DUMP_DIR_ENV))


def _dump_root() -> Path | None:
    dump_dir = os.environ.get(_DUMP_DIR_ENV)
    if not dump_dir:
        return None
    return Path(dump_dir)


def _request_dir(batch) -> Path | None:
    root = _dump_root()
    if root is None or batch is None or getattr(batch, "is_warmup", False):
        return None
    request_id = getattr(batch, "request_id", None) or "request"
    path = root / str(request_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return str(value)


def dump_request_metadata(batch) -> None:
    request_dir = _request_dir(batch)
    if request_dir is None:
        return
    request_id = request_dir.name
    if request_id in _METADATA_WRITTEN:
        return

    metadata = {
        "request_id": request_id,
        "prompt": _to_jsonable(getattr(batch, "prompt", None)),
        "negative_prompt": _to_jsonable(getattr(batch, "negative_prompt", None)),
        "seed": _to_jsonable(getattr(batch, "seed", None)),
        "seeds": _to_jsonable(getattr(batch, "seeds", None)),
        "height": _to_jsonable(getattr(batch, "height", None)),
        "width": _to_jsonable(getattr(batch, "width", None)),
        "num_frames": _to_jsonable(getattr(batch, "num_frames", None)),
        "num_inference_steps": _to_jsonable(
            getattr(batch, "num_inference_steps", None)
        ),
        "guidance_scale": _to_jsonable(getattr(batch, "guidance_scale", None)),
        "guidance_scale_2": _to_jsonable(getattr(batch, "guidance_scale_2", None)),
        "true_cfg_scale": _to_jsonable(getattr(batch, "true_cfg_scale", None)),
        "cfg_normalization": _to_jsonable(getattr(batch, "cfg_normalization", None)),
        "generator_device": _to_jsonable(getattr(batch, "generator_device", None)),
    }
    (request_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )
    _METADATA_WRITTEN.add(request_id)


def _dump_tensor(name: str, tensor: torch.Tensor, batch, max_count: int | None) -> None:
    request_dir = _request_dir(batch)
    if request_dir is None:
        return

    request_id = request_dir.name
    counter_key = (request_id, name)
    count = _DUMP_COUNTS[counter_key]
    if max_count is not None and count >= max_count:
        return

    filename = f"{name}.pt" if max_count == 1 else f"{name}_{count}.pt"
    torch.save(tensor.detach().cpu(), request_dir / filename)
    _DUMP_COUNTS[counter_key] = count + 1


def dump_value(name: str, value: Any, batch=None, max_count: int | None = 1) -> None:
    if not is_tensor_dump_enabled() or value is None:
        return

    dump_request_metadata(batch)
    if isinstance(value, torch.Tensor):
        _dump_tensor(name, value, batch=batch, max_count=max_count)
        return

    if isinstance(value, (list, tuple)):
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if not tensors:
            return
        if len(tensors) == 1:
            _dump_tensor(name, tensors[0], batch=batch, max_count=max_count)
            return
        for idx, tensor in enumerate(tensors):
            _dump_tensor(f"{name}_{idx}", tensor, batch=batch, max_count=max_count)
