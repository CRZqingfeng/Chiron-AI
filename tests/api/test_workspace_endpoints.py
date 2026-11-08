"""工作区出口的三个只读端点：安全边界与内容一致性。

用 FastAPI `TestClient` 走真实 HTTP，不连数据库：这类端点的风险全在
「路径怎么拼」和「越权怎么拦」上，与业务表无关；拖进真实 DB 只会让安全用例
跑得慢、还容易因为环境挂掉。

鉴权用 `dependency_overrides` 换掉 `AuthControl.is_authed`，
`get_accessible_agent` 用 mock 代替——我们要验的是端点自己的边界，
不是权限服务的实现（后者有自己的测试）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.user_agents import workspace as ws_api
from app.core.dependency import AuthControl, DependAuth
from app.settings import settings

USER_ID = 7
AGENT_ID = 1
SESSION = "session_1759900000000"


class _FakeUser:
    id = USER_ID


async def _fake_auth() -> _FakeUser:
    return _FakeUser()


class WorkspaceEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

        patcher = mock.patch.object(settings, "CHAT_WORKSPACE_DIR", str(self.base))
        patcher.start()
        self.addCleanup(patcher.stop)

        accessible = mock.AsyncMock(return_value=object())
        access_patcher = mock.patch.object(ws_api, "get_accessible_agent", accessible)
        access_patcher.start()
        self.addCleanup(access_patcher.stop)
        self.accessible = accessible

        app = FastAPI()
        app.include_router(ws_api.router, prefix="/user-agent", dependencies=[DependAuth])
        app.dependency_overrides[AuthControl.is_authed] = _fake_auth
        self.client = TestClient(app)

        self.ws_dir = ws_api.workspace_root(USER_ID, AGENT_ID, SESSION)
        self.ws_dir.mkdir(parents=True, exist_ok=True)
        # 一律用 write_bytes：Windows 上 write_text 会把 \n 转成 \r\n，
        # 让「下载回来的内容与磁盘一致」这条断言变成在测平台的换行转换
        (self.ws_dir / "report.md").write_bytes("# 报告\n内容\n".encode())
        (self.ws_dir / "notes").mkdir(exist_ok=True)
        (self.ws_dir / "notes" / "a.txt").write_bytes(b"hello\n")
        (self.ws_dir / ".env").write_bytes(b"SECRET=1\n")
        (self.ws_dir / "blob.bin").write_bytes(b"\x00\x01")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _get(self, endpoint: str, **params):
        return self.client.get(
            f"/user-agent/chat/workspace/{endpoint}",
            params={"agent_id": AGENT_ID, "session_id": SESSION, **params},
        )

    # --- 列目录 -----------------------------------------------------------

    def test_list_shows_files_and_hides_dotfiles(self):
        res = self._get("files")
        self.assertEqual(res.status_code, 200)
        names = [it["name"] for it in res.json()["data"]["items"]]
        self.assertIn("report.md", names)
        self.assertIn("notes", names)
        self.assertNotIn(".env", names, "隐藏文件不能展示——里面常是密钥")

    def test_list_missing_workspace_returns_empty(self):
        """还没产出任何文件是正常状态，不该报错。"""
        res = self.client.get(
            "/user-agent/chat/workspace/files",
            params={"agent_id": AGENT_ID, "session_id": "session_other"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["data"]["items"], [])

    def test_list_subdirectory(self):
        res = self._get("files", path="notes")
        names = [it["name"] for it in res.json()["data"]["items"]]
        self.assertEqual(names, ["a.txt"])

    # --- 预览 -------------------------------------------------------------

    def test_preview_text_content(self):
        res = self._get("file", path="report.md")
        self.assertEqual(res.status_code, 200)
        data = res.json()["data"]
        self.assertEqual(data["content"], "# 报告\n内容\n")
        self.assertFalse(data["truncated"])

    def test_preview_rejects_binary_suffix(self):
        """`.bin` 不在预览白名单——用文本接口返回二进制字节，前端会渲染成乱码。"""
        res = self._get("file", path="blob.bin")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], 400)

    def test_preview_missing_file(self):
        res = self._get("file", path="nope.md")
        self.assertEqual(res.json()["code"], 404)

    # --- 下载 -------------------------------------------------------------

    def test_download_returns_same_bytes(self):
        res = self._get("download", path="report.md")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content, "# 报告\n内容\n".encode())
        self.assertIn("attachment", res.headers["content-disposition"])

    def test_download_chinese_filename_uses_rfc5987(self):
        """中文名必须走 `filename*=utf-8''`，否则老浏览器会存成乱码。

        纯 ASCII 名字只会输出 `filename="x.md"`，所以这条必须用中文名才测得到。
        """
        (self.ws_dir / "季度报告.md").write_bytes("# 内容\n".encode())
        res = self._get("download", path="季度报告.md")
        self.assertEqual(res.status_code, 200)
        disposition = res.headers["content-disposition"]
        self.assertIn("filename*=utf-8''", disposition.lower())

    # --- 安全边界（本文件最重要的一组）--------------------------------------

    def test_path_traversal_returns_403(self):
        for bad in ("../../../etc/passwd", "notes/../../x"):
            res = self._get("file", path=bad)
            self.assertEqual(res.status_code, 403, bad)

    def test_url_encoded_traversal_reads_nothing(self):
        """`..%2F..%2F` 会被 HTTP 客户端再编码成字面文件名，落不到穿越语义上。

        它证明不了 403，但必须证明「读不到东西」——这才是要守的底线。
        """
        res = self._get("file", path="..%2F..%2Fsecret.txt")
        self.assertIn(res.status_code, (403, 404))
        self.assertNotIn("root:", res.text)

    def test_absolute_path_returns_403(self):
        res = self._get("file", path=str(self.base / "outside.txt"))
        self.assertEqual(res.status_code, 403)

    def test_traversal_on_list_also_403(self):
        res = self._get("files", path="../..")
        self.assertEqual(res.status_code, 403)

    def test_other_session_cannot_read_foreign_dir(self):
        """跨会话：另一个 session 的目录不在自己的三级目录下，读不到。

        这条是「服务端拼路径」的直接证据——如果哪天有人改成接受前端传 uid，
        这个用例会立刻变红。
        """
        foreign = self.base / str(USER_ID + 1) / str(AGENT_ID) / "session_evil"
        foreign.mkdir(parents=True, exist_ok=True)
        (foreign / "secret.md").write_text("别人的数据\n", encoding="utf-8")

        res = self._get("files", path="")
        names = [it["name"] for it in res.json()["data"]["items"]]
        self.assertNotIn("secret.md", names)

    def test_inaccessible_agent_returns_404(self):
        self.accessible.return_value = None
        res = self._get("files")
        self.assertEqual(res.json()["code"], 404)

    def test_download_rejects_oversized(self):
        big = self.ws_dir / "big.bin"
        big.write_bytes(b"x" * (ws_api.MAX_DOWNLOAD_BYTES + 1))
        res = self._get("download", path="big.bin")
        self.assertEqual(res.status_code, 413)


if __name__ == "__main__":
    unittest.main()
