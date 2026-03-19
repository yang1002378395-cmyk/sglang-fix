import importlib
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="stage-a-cpu-only")
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")

_API_ENV_KEYS = (
    "SGLANG_API_LOGLEVEL",
    "SGLANG_API_LOGDEST",
    "SGLANG_API_DUMP_DIR",
    "SGLANG_API_DUMP_INCLUDE",
    "SGLANG_API_DUMP_EXCLUDE",
)


class _DummyMode:
    def is_idle(self):
        return False

    def is_decode(self):
        return True

    def is_mixed(self):
        return False


class _DummyBatch:
    forward_mode = _DummyMode()


class _DummyLayer:
    tp_q_head_num = 2
    v_head_dim = 4


class _DummyMetadata:
    pass


@contextmanager
def _use_api_logging_env(**envs):
    original_env = {key: os.environ.get(key) for key in _API_ENV_KEYS}

    for key in _API_ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in envs.items():
        os.environ[key] = str(value)

    import sglang.api_logging as api_logging

    try:
        yield importlib.reload(api_logging)
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(api_logging)


def _read_dump_entry(dump_dir: Path) -> Path:
    entries = [path for path in dump_dir.iterdir() if path.is_dir()]
    if len(entries) != 1:
        raise AssertionError(f"Expected exactly one dump entry, got {len(entries)}")
    return entries[0]


def _reload_module(module_name: str):
    module = importlib.import_module(module_name)
    return importlib.reload(module)


def _import_diffusion_module_or_skip(module_name: str):
    try:
        return _reload_module(module_name)
    except Exception as exc:
        raise unittest.SkipTest(f"diffusion import unavailable: {exc}") from exc


class TestAPILogging(CustomTestCase):
    def test_debug_api_disabled_returns_original_function(self):
        with _use_api_logging_env() as api_logging:

            def foo(x):
                return x

            wrapped = api_logging.sglang_debug_api(foo)
            self.assertIs(wrapped, foo)

    def test_level_1_logs_function_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "api.log"
            with _use_api_logging_env(
                SGLANG_API_LOGLEVEL=1,
                SGLANG_API_LOGDEST=log_path,
            ) as api_logging:

                @api_logging.sglang_debug_api
                def foo():
                    return 1

                self.assertEqual(foo(), 1)

            self.assertIn("SGLang API Call: foo", log_path.read_text())

    def test_level_3_logs_tensor_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "api.log"
            with _use_api_logging_env(
                SGLANG_API_LOGLEVEL=3,
                SGLANG_API_LOGDEST=log_path,
            ) as api_logging:

                @api_logging.sglang_debug_api
                def foo(x):
                    return x + 1

                x = torch.randn(2, 3)
                torch.testing.assert_close(foo(x), x + 1)

            log_text = log_path.read_text()
            self.assertIn("SGLang API Call: foo", log_text)
            self.assertIn("shape=(2, 3)", log_text)
            self.assertIn("dtype=torch.float32", log_text)
            self.assertIn("Output:", log_text)

    def test_level_10_dumps_inputs_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            dump_dir = tmpdir_path / "dumps"
            with _use_api_logging_env(
                SGLANG_API_LOGLEVEL=10,
                SGLANG_API_LOGDEST=tmpdir_path / "api.log",
                SGLANG_API_DUMP_DIR=dump_dir,
            ) as api_logging:

                @api_logging.sglang_debug_api
                def foo(x):
                    return x + 1

                x = torch.randn(2, 3)
                torch.testing.assert_close(foo(x), x + 1)

            dump_entry = _read_dump_entry(dump_dir)
            metadata = json.loads((dump_entry / "metadata.json").read_text())
            self.assertEqual(metadata["execution_status"], "completed")
            self.assertTrue((dump_entry / "inputs.pt").exists())
            self.assertTrue((dump_entry / "outputs.pt").exists())

    def test_level_10_preserves_inputs_on_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            dump_dir = tmpdir_path / "dumps"
            with _use_api_logging_env(
                SGLANG_API_LOGLEVEL=10,
                SGLANG_API_LOGDEST=tmpdir_path / "api.log",
                SGLANG_API_DUMP_DIR=dump_dir,
            ) as api_logging:

                @api_logging.sglang_debug_api
                def foo(x):
                    raise RuntimeError("boom")

                with self.assertRaisesRegex(RuntimeError, "boom"):
                    foo(torch.randn(4))

            dump_entry = _read_dump_entry(dump_dir)
            metadata = json.loads((dump_entry / "metadata.json").read_text())
            self.assertEqual(metadata["execution_status"], "exception")
            self.assertEqual(metadata["exception"]["message"], "boom")
            self.assertTrue((dump_entry / "inputs.pt").exists())

    def test_attention_backend_forward_is_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "attn_backend.log"
            with _use_api_logging_env(
                SGLANG_API_LOGLEVEL=1,
                SGLANG_API_LOGDEST=log_path,
            ):
                module = _reload_module("sglang.srt.layers.attention.base_attn_backend")

                class DummyBackend(module.AttentionBackend):
                    def init_forward_metadata(self, forward_batch):
                        raise NotImplementedError

                    def forward_decode(
                        self, q, k, v, layer, forward_batch, save_kv_cache=True
                    ):
                        return q + k + v

                    def forward_extend(
                        self, q, k, v, layer, forward_batch, save_kv_cache=True
                    ):
                        return q + k + v

                backend = DummyBackend()
                x = torch.randn(2, 8)
                torch.testing.assert_close(
                    backend.forward(x, x, x, _DummyLayer(), _DummyBatch()),
                    x + x + x,
                )

            self.assertIn(
                "SGLang API Call: DummyBackend.forward",
                log_path.read_text(),
            )

    def test_multi_platform_forward_is_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "multi_platform.log"
            with _use_api_logging_env(
                SGLANG_API_LOGLEVEL=1,
                SGLANG_API_LOGDEST=log_path,
            ):
                module = _reload_module("sglang.srt.layers.utils.multi_platform")

                class DummyOp(module.MultiPlatformOp):
                    def dispatch_forward(self):
                        return self.forward_native

                    def forward_native(self, x):
                        return x + 1

                op = DummyOp()
                x = torch.randn(2, 3)
                torch.testing.assert_close(op(x), x + 1)

            self.assertIn("SGLang API Call: DummyOp.forward", log_path.read_text())

    def test_diffusion_wrappers_are_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "diffusion.log"
            with _use_api_logging_env(
                SGLANG_API_LOGLEVEL=1,
                SGLANG_API_LOGDEST=log_path,
            ):
                custom_op_module = _import_diffusion_module_or_skip(
                    "sglang.multimodal_gen.runtime.layers.custom_op"
                )
                attn_module = _import_diffusion_module_or_skip(
                    "sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend"
                )

                class DummyCustomOp(custom_op_module.CustomOp):
                    def dispatch_forward(self):
                        return self.forward_native

                    def forward_native(self, x):
                        return x * 2

                class DummyAttentionImpl(attn_module.AttentionImpl):
                    def __init__(
                        self,
                        num_heads: int,
                        head_size: int,
                        softmax_scale: float,
                        causal: bool = False,
                        num_kv_heads: int | None = None,
                        prefix: str = "",
                        **extra_impl_args,
                    ) -> None:
                        self.linear = nn.Identity()

                    def forward(self, query, key, value, attn_metadata):
                        return query + key + value

                x = torch.randn(2, 3)
                op = DummyCustomOp()
                torch.testing.assert_close(op(x), x * 2)

                attn_impl = DummyAttentionImpl(2, 4, 0.5)
                attn_module.wrap_attention_impl_forward(attn_impl)
                torch.testing.assert_close(
                    attn_impl.forward(x, x, x, _DummyMetadata()),
                    x + x + x,
                )

            log_text = log_path.read_text()
            self.assertIn("SGLang API Call: DummyCustomOp.forward", log_text)
            self.assertIn(
                "SGLang API Call: diffusion.attn_impl.DummyAttentionImpl.forward",
                log_text,
            )


if __name__ == "__main__":
    unittest.main()
