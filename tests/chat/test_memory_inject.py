"""U7：记忆注入链路的连通性。

E3（跨会话生效）在 `scripts/verify_long_term_memory.py` 里验的是**数据层**——
`retrieve_user_memory_hits` 在另一个会话里能查到旧会话的事实。这里补的是**链路层**：
预取函数会不会把用户级记忆真的拼进要注入 System 的文本里。

两者缺一不可：数据层通了但预取忘了接，用户还是看不到记忆。
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from app.settings import settings


def _fake_ua() -> SimpleNamespace:
    """只带 `_sub_llm_config_from_ua` 会读的字段。"""
    return SimpleNamespace(
        api_key_ciphertext=None,
        base_url=None,
        model_name="fake-model",
        sub_api_key_ciphertext=None,
        sub_model_name=None,
        sub_base_url=None,
    )


class MemoryInjectTests(unittest.TestCase):
    def _run(self, *, session_hits=None, user_hits=None):
        from app.chat import agent_service

        # 必须是同步函数：真实代码用 asyncio.to_thread(proactive_session_memory_inject_text, ...)
        def _fake_session_inject(*args, **kwargs):
            return session_hits

        with (
            mock.patch(
                "app.chat.memory_search.proactive_session_memory_inject_text",
                side_effect=_fake_session_inject,
            ),
            mock.patch(
                "app.chat.memory_search.retrieve_user_memory_hits",
                return_value=user_hits or [],
            ),
        ):
            return asyncio.run(
                agent_service._prefetch_session_memory(
                    _fake_ua(),
                    "我用什么数据库",
                    user_id=1,
                    agent_id=2,
                    session_id="s1",
                )
            )

    def test_user_memory_injected_alone(self):
        """会话级为空（compact 没触发）时，用户级记忆仍必须注入。

        这是 U7 的关键：会话级依赖 compact（苛刻），用户级依赖 curation（宽松）。
        如果两者写在一个条件里，用户级会被"会话还没到阈值"拖累成空。
        """
        out = self._run(
            session_hits=None,
            user_hits=[{"text": "主力数据库是 MySQL", "kind": "fact", "mem_key": "db"}],
        )
        self.assertIsNotNone(out)
        self.assertIn("长期记忆", out)
        self.assertIn("MySQL", out)

    def test_both_segments_merged_user_first(self):
        out = self._run(
            session_hits="[本会话早期摘录] …",
            user_hits=[{"text": "偏好结论先行", "kind": "preference", "mem_key": "style"}],
        )
        self.assertIn("长期记忆", out)
        self.assertIn("本会话早期摘录", out)
        # 用户级在前：稳定事实比本会话的临时摘录更该被模型优先看到
        self.assertLess(out.index("长期记忆"), out.index("本会话早期摘录"))

    def test_session_only_still_works(self):
        out = self._run(session_hits="[本会话早期摘录] …", user_hits=[])
        self.assertEqual(out, "[本会话早期摘录] …")

    def test_nothing_returns_none(self):
        self.assertIsNone(self._run(session_hits=None, user_hits=[]))

    def test_switch_off_skips_user_memory(self):
        """关掉长期记忆后，即便库里有条目也不该注入。"""
        from app.chat import agent_service

        with (
            mock.patch.object(settings, "CHAT_MEMORY_CURATION_ENABLED", False),
            mock.patch(
                "app.chat.memory_search.proactive_session_memory_inject_text",
                return_value=None,
            ),
            mock.patch(
                "app.chat.memory_search.retrieve_user_memory_hits",
                return_value=[{"text": "不该出现", "kind": "fact", "mem_key": "x"}],
            ),
        ):
            out = asyncio.run(
                agent_service._prefetch_session_memory(
                    _fake_ua(),
                    "q",
                    user_id=1,
                    agent_id=2,
                    session_id="s1",
                )
            )
        self.assertIsNone(out)


class InjectSourceLabelTests(unittest.TestCase):
    """注入来源的标注必须如实。

    背景（A-025 真实数据暴露）：U7 之后 `inj` 可能只有「长期记忆」一段——全新会话
    没有任何被归档的轮次。但思考区原来写死「会话记忆预注入 / 已附加较早轮次摘录」，
    于是用户在一个刚开的会话里看到「已附加较早轮次摘录」，会以为系统在编。
    """

    def test_long_term_only(self):
        from app.chat.memory_search import (
            LONG_TERM_INJECT_MARK,
            format_user_memory_hits,
            inject_sources,
        )

        text = format_user_memory_hits(
            [{"id": 1, "text": "主力数据库是 MySQL", "mem_key": "db", "kind": "fact"}]
        )
        self.assertIn(LONG_TERM_INJECT_MARK, text)
        self.assertEqual(inject_sources(text), ["长期记忆"])

    def test_both_sources_are_reported(self):
        from app.chat.memory_search import (
            LONG_TERM_INJECT_MARK,
            SESSION_EXCERPT_INJECT_MARK,
            inject_sources,
        )

        text = f"{LONG_TERM_INJECT_MARK}事实\n\n{SESSION_EXCERPT_INJECT_MARK}\n[1] 轮次 3：原话"
        self.assertEqual(inject_sources(text), ["长期记忆", "本会话较早摘录"])

    def test_empty_text_has_no_source(self):
        from app.chat.memory_search import inject_sources

        self.assertEqual(inject_sources(""), [])
        self.assertEqual(inject_sources(None), [])  # type: ignore[arg-type]

    def test_label_never_hardcodes_session_excerpt(self):
        """守死文案：源码里不该再出现写死「已附加较早轮次摘录」。"""
        from pathlib import Path as _P

        src = _P("app/chat/agent_service.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "已附加较早轮次摘录",
            src,
            "注入标签必须由 inject_sources() 决定，不能写死——"
            "全新会话里只有长期记忆时，这句话是假的",
        )


if __name__ == "__main__":
    unittest.main()
