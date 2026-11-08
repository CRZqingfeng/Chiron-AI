"""测试命令授权存储与会话作用域（U10 P1-E）。

Redis 用替身：这层要验的是「作用域为空 / Redis 挂了都不能变成放行」，
而不是 Redis 本身。方向永远保守——多问一次，而不是静默放行。
"""

from __future__ import annotations

import unittest
from unittest import mock

from app.core.turn_runtime import get_session_scope, make_session_scope, set_session_scope
from app.workspace import command_grants as cg

SCOPE = "u1_a2_ssession_1762500000000"


class GrantStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store: dict[str, object] = {}

        def _set(key, value, ttl=None):  # noqa: ANN001 - 替身签名跟 cache 对齐
            self.store[key] = value
            return True

        def _get(key):  # noqa: ANN001
            return self.store.get(key)

        def _delete(key):  # noqa: ANN001
            self.store.pop(key, None)
            return None

        for name, fn in (
            ("set_json", _set),
            ("get_json", _get),
            ("delete", _delete),
        ):
            p = mock.patch.object(cg.cache, name, side_effect=fn)
            p.start()
            self.addCleanup(p.stop)

    def test_grant_then_granted(self):
        self.assertFalse(cg.test_commands_granted(SCOPE))
        self.assertTrue(cg.grant_test_commands(SCOPE))
        self.assertTrue(cg.test_commands_granted(SCOPE))

    def test_scopes_are_isolated(self):
        """换会话就要重新确认——授权不能跨会话泄漏。"""
        cg.grant_test_commands(SCOPE)
        self.assertFalse(cg.test_commands_granted("u1_a2_sother"))

    def test_empty_scope_never_grants_or_writes(self):
        self.assertFalse(cg.grant_test_commands(""))
        self.assertFalse(cg.grant_test_commands(None))
        self.assertFalse(cg.test_commands_granted(""))
        self.assertFalse(cg.test_commands_granted(None))
        self.assertEqual(self.store, {}, "空作用域不该写任何东西")

    def test_revoke(self):
        cg.grant_test_commands(SCOPE)
        cg.revoke_test_commands(SCOPE)
        self.assertFalse(cg.test_commands_granted(SCOPE))

    def test_redis_read_failure_is_conservative(self):
        cg.grant_test_commands(SCOPE)
        with mock.patch.object(cg.cache, "get_json", side_effect=RuntimeError("redis down")):
            self.assertFalse(
                cg.test_commands_granted(SCOPE), "Redis 挂了必须按「未授权」处理"
            )

    def test_redis_write_failure_reported(self):
        with mock.patch.object(cg.cache, "set_json", side_effect=RuntimeError("redis down")):
            self.assertFalse(cg.grant_test_commands(SCOPE))

    def test_malformed_payload_is_not_a_grant(self):
        self.store[cg._key(SCOPE)] = {"granted": False}
        self.assertFalse(cg.test_commands_granted(SCOPE))
        self.store[cg._key(SCOPE)] = "garbage"
        self.assertFalse(cg.test_commands_granted(SCOPE))


class SessionScopeTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_session_scope(None)

    def test_scope_contains_all_three_parts(self):
        scope = make_session_scope(7, 3, "session_abc")
        self.assertEqual(scope, "u7_a3_ssession_abc")

    def test_scope_is_sanitized_for_redis_keys(self):
        """会话 id 来自前端，可能含 `/`、`..`，不能原样进 key。"""
        self.assertEqual(make_session_scope(1, 2, "a/../b"), "u1_a2_sa____b")

    def test_set_and_get(self):
        self.assertIsNone(get_session_scope())
        set_session_scope("u1_a2_sx")
        self.assertEqual(get_session_scope(), "u1_a2_sx")
        set_session_scope("")
        self.assertIsNone(get_session_scope(), "空串要归一成 None，否则会写出空 key")


if __name__ == "__main__":
    unittest.main()
