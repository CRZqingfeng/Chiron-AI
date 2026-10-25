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


if __name__ == "__main__":
    unittest.main()
