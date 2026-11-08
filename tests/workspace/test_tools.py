"""工作区工具层：注册、往返、错误语义、以及「execute 不被暴露」这条设计约束。

用 LocalWorkspace 跑，不启容器——工具层与后端无关，容器行为已在 test_docker 覆盖。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.workspace.local import LocalWorkspace
from app.workspace.tools import make_workspace_tools


class WorkspaceToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))
        self.tools = {t.name: t for t in make_workspace_tools(self.ws)}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # --- 注册面 -----------------------------------------------------------

    def test_exactly_four_tools_registered(self):
        """U10 P0-A 加了 `workspace_edit_file`，所以是四个。

        `grep`/`glob` 在 P0-B 加进来后会变成六个——届时这条断言要跟着改，
        这是刻意的：工具面发生变化必须有人看一眼。
        """
        self.assertEqual(
            sorted(self.tools),
            [
                "workspace_edit_file",
                "workspace_list_dir",
                "workspace_read_file",
                "workspace_write_file",
            ],
        )

    def test_execute_is_deliberately_not_exposed(self):
        """U4 只暴露文件类工具。`execute` 接口已实现，但要等 U5b 的审批中断
        一起上线——否则模型可以直接跑任意 shell，中间存在无保护窗口。
        这条测试就是守这个边界的：谁哪天顺手加了 run_command，这里会红。
        """
        names = set(self.tools)
        self.assertNotIn("workspace_run_command", names)
        self.assertNotIn("workspace_exec", names)
        self.assertNotIn("run_command", names)
        for name in names:
            self.assertNotIn("exec", name.split("_"))

    def test_tool_names_are_prefixed(self):
        """统一前缀让 `_sort_tools`（按名字排序）把它们聚在一起。"""
        for name in self.tools:
            self.assertTrue(name.startswith("workspace_"), name)

    # --- 往返 -------------------------------------------------------------

    def test_write_then_read(self):
        w = self.tools["workspace_write_file"].invoke({"path": "a.txt", "content": "第一行"})
        self.assertIn("已写入", w)
        r = self.tools["workspace_read_file"].invoke({"path": "a.txt"})
        self.assertEqual(r, "第一行")

    def test_write_creates_dirs(self):
        self.tools["workspace_write_file"].invoke({"path": "x/y/z.md", "content": "v"})
        self.assertTrue((Path(self._tmp.name) / "x/y/z.md").exists())

    def test_list_dir_shows_entries(self):
        self.tools["workspace_write_file"].invoke({"path": "note.md", "content": "a"})
        (Path(self._tmp.name) / "sub").mkdir()
        out = self.tools["workspace_list_dir"].invoke({})
        self.assertIn("note.md", out)
        self.assertIn("sub", out)
        self.assertIn("目录", out)

    def test_list_empty_dir(self):
        out = self.tools["workspace_list_dir"].invoke({"path": "."})
        self.assertIn("为空", out)

    # --- 错误语义：返回字符串而不是抛异常 ---------------------------------

    def test_read_missing_returns_error_string(self):
        out = self.tools["workspace_read_file"].invoke({"path": "nope.txt"})
        self.assertTrue(out.startswith("错误："), out)
        self.assertIn("不存在", out)

    def test_path_escape_returns_error_string(self):
        """工具层必须把越界路径变成可读错误，而不是让异常冒泡中断整轮生成。"""
        out = self.tools["workspace_write_file"].invoke({"path": "../hack.txt", "content": "x"})
        self.assertTrue(out.startswith("错误："), out)
        self.assertFalse((Path(self._tmp.name).parent / "hack.txt").exists())

    def test_read_dir_returns_error_string(self):
        (Path(self._tmp.name) / "d").mkdir()
        out = self.tools["workspace_read_file"].invoke({"path": "d"})
        self.assertTrue(out.startswith("错误："), out)

    # --- 截断 -------------------------------------------------------------

    def test_read_truncates_long_text(self):
        self.tools["workspace_write_file"].invoke({"path": "big.txt", "content": "x" * 5000})
        out = self.tools["workspace_read_file"].invoke({"path": "big.txt", "max_chars": 1000})
        self.assertIn("已截断", out)
        self.assertLess(len(out), 1200)

    def test_read_empty_file(self):
        self.tools["workspace_write_file"].invoke({"path": "e.txt", "content": ""})
        self.assertEqual(self.tools["workspace_read_file"].invoke({"path": "e.txt"}), "(文件为空)")

    def test_invalid_max_chars_rejected_by_schema(self):
        """非法类型在工具入口就被 pydantic schema 拒绝，函数体内的兜底拿不到。

        这比「静默钳值」更好：错误在边界暴露，LangChain 的 ToolNode 会把它转成
        一条 ToolMessage 回给模型，模型据此自我纠正，而不会让整轮生成中断。
        函数体内仍保留 clamp（见 tools.py），那是给直接调 `.func()` 的场景兜底。
        """
        self.tools["workspace_write_file"].invoke({"path": "z.txt", "content": "hello"})
        with self.assertRaises(Exception):
            self.tools["workspace_read_file"].invoke({"path": "z.txt", "max_chars": "abc"})

    def test_func_level_clamp_still_works(self):
        """绕过 schema 直接调函数体时，非法值应被钳到合法区间而不是崩。"""
        self.tools["workspace_write_file"].invoke({"path": "y.txt", "content": "hello"})
        out = self.tools["workspace_read_file"].func("y.txt", "abc")
        self.assertEqual(out, "hello")

    # --- edit_file：返回形状是 ACI 的一部分 ---------------------------------

    def test_edit_returns_diff_context(self):
        """成功返回要带行号与改动前后各 3 行，模型才能自检而不用再读全文。"""
        self.tools["workspace_write_file"].invoke(
            {"path": "a.py", "content": "def f():\n    return 1\n\nprint(f())\n"}
        )
        out = self.tools["workspace_edit_file"].invoke(
            {"path": "a.py", "old_string": "    return 1", "new_string": "    return 2"}
        )
        self.assertIn("已修改 a.py", out)
        self.assertIn("第 2 行", out)
        self.assertIn("改动前", out)
        self.assertIn("改动后", out)
        self.assertIn("|     return 2", out)
        self.assertNotIn("|     return 1", out.split("改动后")[-1])

    def test_edit_not_found_suggests_reading_first(self):
        """只说「未找到」模型会原地重试；必须给出下一步动作。"""
        self.tools["workspace_write_file"].invoke({"path": "a.txt", "content": "hello\n"})
        out = self.tools["workspace_edit_file"].invoke(
            {"path": "a.txt", "old_string": "missing", "new_string": "x"}
        )
        self.assertIn("未找到匹配内容", out)
        self.assertIn("workspace_read_file", out)

    def test_edit_ambiguous_lists_lines_and_mentions_replace_all(self):
        self.tools["workspace_write_file"].invoke({"path": "a.txt", "content": "foo\nbar\nfoo\n"})
        out = self.tools["workspace_edit_file"].invoke(
            {"path": "a.txt", "old_string": "foo", "new_string": "baz"}
        )
        self.assertIn("2 处匹配", out)
        self.assertIn("第 1、3 行", out)
        self.assertIn("replace_all", out)


if __name__ == "__main__":
    unittest.main()
