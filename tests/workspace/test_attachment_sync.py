"""附件镜像到工作区（U10 P1-D）。

要守的三条：落盘不阻断上传、配额到了要跳过、清理要跟会话删除一起走。
另有一条容易忽略的：注入给模型的提示必须**由内容决定**——目录里没文件就不能说
「已同步到 _attachments/」，否则就是系统在编。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.settings import settings
from app.workspace.attachment_sync import (
    ATTACHMENT_DIR,
    MAX_MIRROR_BYTES,
    attachment_mirror_hint,
    attachment_mirror_names,
    purge_workspace_attachments,
    purge_workspace_attachments_for_agent,
    sync_attachment_to_workspace,
)
from app.workspace.base import safe_resolve
from app.workspace import workspace_root

USER_ID = 1
AGENT_ID = 2
SESSION = "session_1762500000000"


class AttachmentMirrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        for target, value in (
            ("CHAT_WORKSPACE_DIR", str(self.base)),
            ("CHAT_WORKSPACE_ENABLED", True),
        ):
            p = mock.patch.object(settings, target, value)
            p.start()
            self.addCleanup(p.stop)
        self.ws_dir = workspace_root(USER_ID, AGENT_ID, SESSION)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _mirror(self, name: str = "data.csv", raw: bytes = b"a,b\n1,2\n") -> str | None:
        return sync_attachment_to_workspace(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            session_id=SESSION,
            filename=name,
            raw=raw,
        )

    # --- 正常路径 ---------------------------------------------------------

    def test_writes_file_into_session_workspace(self):
        rel = self._mirror()
        self.assertEqual(rel, f"{ATTACHMENT_DIR}/data.csv")
        target = self.ws_dir / ATTACHMENT_DIR / "data.csv"
        self.assertTrue(target.exists())
        self.assertEqual(target.read_bytes(), b"a,b\n1,2\n")

    def test_keeps_chinese_filename(self):
        """文件名是给人看的，中文不该被替换成下划线。"""
        rel = self._mirror("季度数据 2026.csv")
        self.assertEqual(rel, f"{ATTACHMENT_DIR}/季度数据 2026.csv")
        self.assertTrue((self.ws_dir / ATTACHMENT_DIR / "季度数据 2026.csv").exists())

    def test_same_name_overwrites(self):
        self._mirror("a.txt", b"old")
        self._mirror("a.txt", b"new")
        target = self.ws_dir / ATTACHMENT_DIR / "a.txt"
        self.assertEqual(target.read_bytes(), b"new")

    def test_path_in_filename_is_stripped(self):
        """上传文件名里的 `..` 不能变成目录穿越。"""
        rel = self._mirror("../../evil.txt")
        self.assertEqual(rel, f"{ATTACHMENT_DIR}/evil.txt")
        self.assertFalse((self.base / "evil.txt").exists())

    # --- 边界 -------------------------------------------------------------

    def test_skips_when_over_quota(self):
        big = b"x" * (MAX_MIRROR_BYTES + 1)
        self.assertIsNone(self._mirror("huge.bin", big))
        self.assertFalse((self.ws_dir / ATTACHMENT_DIR / "huge.bin").exists())

    def test_replacing_same_file_does_not_double_count_quota(self):
        """同名覆盖时，原文件占的空间要让出来，否则重传几次就撞配额。"""
        chunk = b"x" * (MAX_MIRROR_BYTES // 2)
        self.assertIsNotNone(self._mirror("a.bin", chunk + b"1"))
        self.assertIsNotNone(self._mirror("a.bin", chunk + b"2"))

    def test_noop_when_workspace_disabled(self):
        with mock.patch.object(settings, "CHAT_WORKSPACE_ENABLED", False):
            self.assertIsNone(self._mirror())
        self.assertFalse((self.ws_dir / ATTACHMENT_DIR).exists())

    def test_empty_payload_is_skipped(self):
        self.assertIsNone(self._mirror("empty.txt", b""))

    # --- 提示：必须由内容决定 ---------------------------------------------

    def test_hint_empty_before_any_file(self):
        self.assertEqual(attachment_mirror_hint(USER_ID, AGENT_ID, SESSION), "")

    def test_hint_mentions_real_files(self):
        self._mirror("data.csv")
        hint = attachment_mirror_hint(USER_ID, AGENT_ID, SESSION)
        self.assertIn(ATTACHMENT_DIR, hint)
        self.assertIn("data.csv", hint)
        self.assertIn("workspace_run_command", hint)

    def test_hint_empty_when_workspace_disabled(self):
        self._mirror("data.csv")
        with mock.patch.object(settings, "CHAT_WORKSPACE_ENABLED", False):
            self.assertEqual(attachment_mirror_hint(USER_ID, AGENT_ID, SESSION), "")

    # --- 清理 -------------------------------------------------------------

    def test_purge_removes_mirror_dir(self):
        self._mirror("a.csv")
        self._mirror("b.csv")
        purge_workspace_attachments(USER_ID, AGENT_ID, SESSION)
        self.assertFalse((self.ws_dir / ATTACHMENT_DIR).exists())
        # 工作区本身要留着——里面可能还有 agent 写的其他文件
        self.assertTrue(self.ws_dir.exists())

    def test_purge_keeps_other_workspace_files(self):
        self._mirror("a.csv")
        report = self.ws_dir / "report.md"
        report.write_bytes(b"# r")
        purge_workspace_attachments(USER_ID, AGENT_ID, SESSION)
        self.assertTrue(report.exists())

    def test_purge_for_agent_walks_all_sessions(self):
        other = workspace_root(USER_ID, AGENT_ID, "session_other")
        for session in (SESSION, "session_other"):
            sync_attachment_to_workspace(
                user_id=USER_ID,
                agent_id=AGENT_ID,
                session_id=session,
                filename="x.csv",
                raw=b"1",
            )
        purge_workspace_attachments_for_agent(USER_ID, AGENT_ID)
        self.assertFalse((self.ws_dir / ATTACHMENT_DIR).exists())
        self.assertFalse((other / ATTACHMENT_DIR).exists())

    def test_names_listing(self):
        self.assertEqual(attachment_mirror_names(USER_ID, AGENT_ID, SESSION), [])
        self._mirror("b.csv")
        self._mirror("a.csv")
        self.assertEqual(attachment_mirror_names(USER_ID, AGENT_ID, SESSION), ["a.csv", "b.csv"])

    def test_safe_resolve_agrees_with_target(self):
        """镜像必须落在会话工作区内——这条防的是将来有人把根目录改错。"""
        self._mirror()
        target = safe_resolve(self.ws_dir, f"{ATTACHMENT_DIR}/data.csv")
        self.assertTrue(target.is_relative_to(self.ws_dir.resolve()))


if __name__ == "__main__":
    unittest.main()
