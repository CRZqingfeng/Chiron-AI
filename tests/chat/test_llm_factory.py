"""LlmFactory：统一的模型构造入口。

本文件锁定的不是"能构造出模型"，而是**参数不被悄悄改动**：
temperature=0 不能被丢掉、temperature=None 不能被当成 0、
缺 Key 时 chat 抛异常而 try_chat 返回 None。
这些点一旦错了，运行时不会报错，只会在某条辅助链路上悄悄改变行为。
"""

from __future__ import annotations

import unittest
from unittest import mock

from app.utils import llm_factory
from app.utils.llm_factory import LlmConfigError, LlmFactory

_CFG = {
    "api_key": "  sk-test-123  ",
    "base_url": " https://api.test.example/v1 ",
    "model_name": " test-model ",
}


class _Recorder:
    """替换 init_chat_model，记录 kwargs 并返回哨兵对象。"""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return "MODEL"


def _patched():
    rec = _Recorder()
    return rec, mock.patch.object(llm_factory, "init_chat_model", rec)


class LlmFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        # 避免真实构建 httpx 客户端/DNS：只验证 base_url 是否被传给 pinned 函数
        self._pinned = mock.patch.object(
            llm_factory,
            "pinned_llm_client_kwargs",
            side_effect=lambda url: {"http_client": "SYNC", "http_async_client": "ASYNC"} if url else {},
        )
        self._pinned.start()

    def tearDown(self) -> None:
        self._pinned.stop()

    def test_strips_api_key_base_url_model(self):
        rec, patch = _patched()
        with patch:
            LlmFactory.chat(_CFG)
        self.assertEqual(rec.kwargs["api_key"], "sk-test-123")
        self.assertEqual(rec.kwargs["base_url"], "https://api.test.example/v1")
        self.assertEqual(rec.kwargs["model"], "test-model")

    def test_missing_api_key_raises_in_chat(self):
        with self.assertRaises(LlmConfigError) as ctx:
            LlmFactory.chat({"api_key": "   ", "model_name": "m"})
        self.assertIn("API Key", str(ctx.exception))

    def test_none_cfg_raises_in_chat(self):
        with self.assertRaises(LlmConfigError):
            LlmFactory.chat(None)

    def test_missing_api_key_returns_none_in_try_chat(self):
        self.assertIsNone(LlmFactory.try_chat({"api_key": "", "model_name": "m"}))
        self.assertIsNone(LlmFactory.try_chat(None))

    def test_default_model_name_is_gpt4(self):
        rec, patch = _patched()
        with patch:
            LlmFactory.chat({"api_key": "sk-x"})
        self.assertEqual(rec.kwargs["model"], "gpt-4")

    def test_model_name_override_wins(self):
        rec, patch = _patched()
        with patch:
            LlmFactory.chat(_CFG, model_name="grade-model")
        self.assertEqual(rec.kwargs["model"], "grade-model")

    def test_temperature_none_is_omitted(self):
        """vision_caption 依赖 SDK 默认 0.7，必须完全不传该键。"""
        rec, patch = _patched()
        with patch:
            LlmFactory.chat(_CFG, temperature=None)
        self.assertNotIn("temperature", rec.kwargs)

    def test_temperature_zero_is_passed(self):
        """0 是有效值，不能被当成假值丢掉。"""
        rec, patch = _patched()
        with patch:
            LlmFactory.chat(_CFG, temperature=0)
        self.assertIn("temperature", rec.kwargs)
        self.assertEqual(rec.kwargs["temperature"], 0)

    def test_stream_usage_default_and_explicit(self):
        rec, patch = _patched()
        with patch:
            LlmFactory.chat(_CFG)
        self.assertIs(rec.kwargs["stream_usage"], False)

        rec2, patch2 = _patched()
        with patch2:
            LlmFactory.chat(_CFG, stream_usage=True)
        self.assertIs(rec2.kwargs["stream_usage"], True)

    def test_stream_usage_none_is_omitted(self):
        """vision 依赖 langchain 的默认用量策略，必须完全不传该键。"""
        rec, patch = _patched()
        with patch:
            LlmFactory.chat(_CFG, stream_usage=None)
        self.assertNotIn("stream_usage", rec.kwargs)

    def test_overrides_pass_through(self):
        rec, patch = _patched()
        with patch:
            LlmFactory.chat(_CFG, timeout=60, max_tokens=1, max_retries=3)
        self.assertEqual(rec.kwargs["timeout"], 60)
        self.assertEqual(rec.kwargs["max_tokens"], 1)
        self.assertEqual(rec.kwargs["max_retries"], 3)

    def test_pinned_client_only_when_base_url(self):
        rec, patch = _patched()
        with patch:
            LlmFactory.chat(_CFG)
        self.assertEqual(rec.kwargs["http_client"], "SYNC")

        rec2, patch2 = _patched()
        with patch2:
            LlmFactory.chat({"api_key": "sk-x", "model_name": "m", "base_url": "   "})
        self.assertNotIn("http_client", rec2.kwargs)
        self.assertIsNone(rec2.kwargs["base_url"])

    def test_explicit_temperature_beats_override(self):
        """即使 overrides 里误传 temperature，显式参数也应生效。"""
        rec, patch = _patched()
        with patch:
            LlmFactory.chat(_CFG, temperature=0.2, **{"temperature_ignored": 1})
        self.assertEqual(rec.kwargs["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
