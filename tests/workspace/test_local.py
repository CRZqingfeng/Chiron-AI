"""LocalWorkspace：路径校验、读写、列目录、命令执行与截断。

全部用 `tempfile` 建真实目录（不 mock 文件系统）——路径逃逸这类问题只有真跑才能发现。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from app.workspace.base import PathEscapeError, WorkspaceError, safe_resolve, truncate_text
from app.workspace.local import LocalWorkspace


class SafeResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_normal_relative_path(self):
        got = safe_resolve(self.root, "a/b.txt")
        self.assertEqual(got, (self.root / "a/b.txt").resolve())

    def test_rejects_parent_traversal(self):
        for bad in ("../x", "a/../../x", "..", "./../x"):
            with self.subTest(path=bad), self.assertRaises(PathEscapeError):
                safe_resolve(self.root, bad)

    def test_rejects_absolute_paths(self):
        """`root / "/etc/passwd"` 在 pathlib 下会直接变成 `/etc/passwd`，必须显式挡。"""
        cases = ["/etc/passwd"]
        if os.name == "nt":
            cases += ["C:\\Windows\\win.ini", "D:/x"]
        for bad in cases:
            with self.subTest(path=bad), self.assertRaises(PathEscapeError):
                safe_resolve(self.root, bad)

    def test_rejects_empty(self):
        with self.assertRaises(PathEscapeError):
            safe_resolve(self.root, "   ")

    def test_rejects_symlink_escape(self):
        """工作区内指向外部的软链，resolve 后落在根外 → 必须拒绝。

        ⚠️ Windows 下这一条**在当前机器上无法验证**：无 `SeCreateSymbolicLinkPrivilege`
        时 `symlink_to` 既不抛异常也不创建文件（实测 `exists()=False`、
        `lstat` 报 FileNotFoundError），因此只能跳过，不能在 Windows 上证明行为。
        Linux（生产环境）上 `resolve()` 会解链，`is_relative_to` 随之拒绝。
        """
        outside = tempfile.mkdtemp()
        try:
            link = self.root / "link_out"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("当前环境不允许创建符号链接")
            # 关键：不能只看有没有抛异常——Windows 会静默失败
            if not link.is_symlink():
                self.skipTest("当前环境 symlink_to 静默失败（Windows 无特权）")
            with self.assertRaises(PathEscapeError):
                safe_resolve(self.root, "link_out/secret.txt")
        finally:
            import shutil

            shutil.rmtree(outside, ignore_errors=True)


class LocalWorkspaceFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = LocalWorkspace(self.root)

    def tearDown(self) -> None:
        self.ws.cleanup()
        self._tmp.cleanup()

    def test_write_read_roundtrip(self):
        written = self.ws.write_text("a.txt", "你好\nworld")
        self.assertEqual(written, len("你好\nworld".encode()))
        self.assertEqual(self.ws.read_text("a.txt"), "你好\nworld")

    def test_write_creates_parent_dirs(self):
        self.ws.write_text("deep/nested/c.txt", "x")
        self.assertTrue((self.root / "deep/nested/c.txt").exists())

    def test_read_missing_file_raises(self):
        with self.assertRaises(WorkspaceError):
            self.ws.read_text("nope.txt")

    def test_read_dir_raises_with_hint(self):
        (self.root / "d").mkdir()
        with self.assertRaises(WorkspaceError) as ctx:
            self.ws.read_text("d")
        self.assertIn("list_dir", str(ctx.exception))

    def test_read_truncates_by_bytes(self):
        """超限时保留前缀并明确告知，模型才知道自己看到的不是全部。"""
        self.ws.write_text("big.txt", "a" * 500)
        got = self.ws.read_text("big.txt", max_bytes=100)
        self.assertTrue(got.startswith("a" * 100))
        self.assertIn("仅显示前 100 字节", got)
        self.assertLess(len(got), 500)

    def test_write_into_subdir_then_list(self):
        self.ws.write_text("x/y.txt", "1")
        self.ws.write_text("z.txt", "2")
        items = self.ws.list_dir(".")
        names = {i["path"] for i in items}
        # 目录排在文件前，且路径是相对 + POSIX 分隔符
        self.assertEqual(names, {"x", "z.txt"})
        dirs = [i for i in items if i["is_dir"]]
        self.assertEqual([d["path"] for d in dirs], ["x"])
        self.assertEqual(items[0]["path"], "x")

    def test_list_dir_missing_raises(self):
        with self.assertRaises(WorkspaceError):
            self.ws.list_dir("nope")

    def test_cleanup_is_idempotent_and_keeps_files(self):
        self.ws.write_text("keep.txt", "v")
        self.ws.cleanup()
        self.ws.cleanup()
        self.assertTrue((self.root / "keep.txt").exists(), "cleanup 不应删除工作区文件")

    def test_workspace_escape_via_write_rejected(self):
        with self.assertRaises(PathEscapeError):
            self.ws.write_text("../escape.txt", "x")


class LocalWorkspaceExecTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = LocalWorkspace(self.root, exec_timeout=30.0, output_limit=1000)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_execute_ok(self):
        cmd = "echo hi" if os.name != "nt" else "echo hi"
        r = self.ws.execute(cmd)
        self.assertEqual(r["exit_code"], 0)
        self.assertIn("hi", r["stdout"])
        self.assertFalse(r["truncated"])

    def test_execute_nonzero_is_not_exception(self):
        """非 0 退出码是正常结果，不能抛异常——否则模型无法从错误里学习。"""
        r = self.ws.execute("exit 3", timeout=5)
        self.assertEqual(r["exit_code"], 3)

    def test_execute_timeout(self):
        # 用当前解释器睡 5 秒：跨平台，且不依赖 PATH 上的 `timeout`。
        #
        # 原来 Windows 分支写的是 `timeout /t 5 /nobreak`，在 Git Bash 里必红：
        # shell=True 走 cmd.exe，而 cmd.exe 继承了 Git Bash 的 PATH（/usr/bin 在
        # system32 之前），于是 `timeout` 解析到 GNU coreutils 版——它不认 `/t`，
        # 报 "invalid time interval" 并返回 **125**，与期望的 -1（我们自己判的超时）
        # 完全不同。从 cmd/PowerShell 启动时又会通过，属于「换个启动方式就变脸」。
        cmd = f'"{sys.executable}" -c "import time; time.sleep(5)"'
        r = self.ws.execute(cmd, timeout=0.5)
        self.assertEqual(r["exit_code"], -1)
        self.assertIn("超时", r["stderr"])
        self.assertTrue(r["truncated"] is False)

    def test_timeout_returns_promptly(self):
        """超时必须**及时**返回：它是失控命令的刹车，慢刹车等于没刹车。

        曾经的缺陷：`subprocess.run(timeout=)` 只 kill shell，孙进程继续持有管道，
        `communicate` 要等它自然退出 —— 设 0.5s 超时实际等满 5.8s。
        现在 `_kill_process_tree` 杀整棵树（Windows `taskkill /F /T`、POSIX `killpg`）。
        这条断言就是那次的回归守卫：改回 `subprocess.run` 会立刻红。
        """
        cmd = f'"{sys.executable}" -c "import time; time.sleep(5)"'
        r = self.ws.execute(cmd, timeout=0.5)
        self.assertLess(r["duration_ms"], 4000)

    def test_execute_output_truncated(self):
        # 输出 5000 字符，limit=1000 → 应截断
        cmd = "python -c \"print('x'*5000)\"" if sys.platform != "win32" else "python -c \"print('x'*5000)\""
        r = self.ws.execute(cmd, timeout=15)
        self.assertTrue(r["truncated"])
        self.assertIn("输出已截断", r["stdout"])
        self.assertLess(len(r["stdout"]), 2000)

    def test_execute_cwd_escape_rejected(self):
        with self.assertRaises(PathEscapeError):
            self.ws.execute("echo x", cwd="..")

    def test_execute_cwd_must_exist(self):
        with self.assertRaises(WorkspaceError):
            self.ws.execute("echo x", cwd="nodir")

    def test_execute_runs_in_workspace(self):
        """cwd 锁定在工作区：命令产生的文件应落在工作区内。"""
        self.ws.execute("echo hello > out.txt")
        self.assertTrue((self.root / "out.txt").exists())


class TruncateTextTests(unittest.TestCase):
    """截断必须保头尾（U10 P1-C）。

    依据是 ACI 的基本判断：命令输出里**动作在开头、错误在最后几行，中间才是噪音**。
    只保头会把 pytest 的失败摘要、编译器的 error 行整段切掉——那是模型唯一的反馈信号。
    """

    def test_keeps_tail_where_errors_live(self):
        text = "START-MARK\n" + "x" * 5000 + "\nFAILED test_foo.py::test_bar - AssertionError"
        out, truncated = truncate_text(text, 1000)
        self.assertTrue(truncated)
        self.assertIn("START-MARK", out, "开头要保住——动作与摘要在这里")
        self.assertIn("AssertionError", out, "结尾要保住——错误在这里")
        self.assertIn("输出已截断", out)

    def test_reports_omitted_count(self):
        out, _ = truncate_text("a" * 5000, 1000)
        self.assertIn("省略中间", out)
        self.assertIn("全文 5000 字符", out)

    def test_short_text_untouched(self):
        text = "short"
        out, truncated = truncate_text(text, 1000)
        self.assertFalse(truncated)
        self.assertEqual(out, text)

    def test_zero_limit_means_no_truncation(self):
        text = "abc" * 1000
        out, truncated = truncate_text(text, 0)
        self.assertFalse(truncated)
        self.assertEqual(out, text)

    def test_result_not_much_longer_than_limit(self):
        """尾注不能把结果撑爆：多出来的只应是那几十字说明。"""
        out, _ = truncate_text("a" * 100_000, 1000)
        self.assertLess(len(out), 1200)


if __name__ == "__main__":
    unittest.main()
