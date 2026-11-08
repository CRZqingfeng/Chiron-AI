"""工作区工具层：注册、往返、错误语义、以及「execute 不被暴露」这条设计约束。

用 LocalWorkspace 跑，不启容器——工具层与后端无关，容器行为已在 test_docker 覆盖。
"""

from __future__ import annotations

import re
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

    def test_exactly_six_tools_registered(self):
        """六个：list / read / write / edit / grep / glob。

        数量断言是刻意写的护栏——工具面变化必须有人看一眼，而不是悄悄多出来。
        """
        self.assertEqual(
            sorted(self.tools),
            [
                "workspace_edit_file",
                "workspace_glob",
                "workspace_grep",
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
        # U10 P1-A 起 read_file 带行号（ACI：模型要能引用「第几行」）
        self.assertEqual(r, "  1 | 第一行")

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

    def test_read_single_huge_line_is_clipped(self):
        """单行 5000 字符（压缩 JSON / 单行 CSV）：不能整行吐出来，也不能返回空。

        行内没法续读，所以提示必须引导模型换策略（grep 定位），
        而不是给一个会跳过半行的 offset。
        """
        self.tools["workspace_write_file"].invoke({"path": "big.txt", "content": "x" * 5000})
        out = self.tools["workspace_read_file"].invoke({"path": "big.txt", "max_chars": 1000})
        self.assertLess(len(out), 1400)
        self.assertIn("已截断", out)
        self.assertIn("workspace_grep", out)

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
        self.assertIn("hello", out)

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

    # --- grep / glob：只读工具 ---------------------------------------------

    def test_grep_returns_paths_and_lines(self):
        self.tools["workspace_write_file"].invoke({"path": "a.py", "content": "x = 1\nfoo = 2\n"})
        self.tools["workspace_write_file"].invoke({"path": "b.txt", "content": "foo\n"})
        out = self.tools["workspace_grep"].invoke({"pattern": "foo"})
        self.assertIn("a.py:2:", out)
        self.assertIn("b.txt:1:", out)

    def test_grep_hints_regex_when_metachars_and_no_hit(self):
        """静默不命中比报错更糟：模型会以为文件里没有，然后换方法乱试。"""
        self.tools["workspace_write_file"].invoke({"path": "a.py", "content": "x = 1\n"})
        out = self.tools["workspace_grep"].invoke({"pattern": "foo|bar"})
        self.assertIn("没有找到", out)
        self.assertIn("regex=True", out)

    def test_glob_lists_files(self):
        self.tools["workspace_write_file"].invoke({"path": "src/a.py", "content": "x"})
        self.tools["workspace_write_file"].invoke({"path": "src/b.md", "content": "y"})
        out = self.tools["workspace_glob"].invoke({"pattern": "**/*.py"})
        self.assertIn("src/a.py", out)
        self.assertNotIn("b.md", out)

    # --- read 分页（U10 P1-A）---------------------------------------------

    def test_read_has_line_numbers(self):
        content = "\n".join(f"line{i}" for i in range(1, 6))
        self.tools["workspace_write_file"].invoke({"path": "a.txt", "content": content})
        out = self.tools["workspace_read_file"].invoke({"path": "a.txt"})
        self.assertIn("  1 | line1", out)
        self.assertIn("  5 | line5", out)

    def test_read_offset_is_one_based(self):
        content = "\n".join(f"line{i}" for i in range(1, 11))
        self.tools["workspace_write_file"].invoke({"path": "a.txt", "content": content})
        out = self.tools["workspace_read_file"].invoke({"path": "a.txt", "offset": 3, "limit": 2})
        self.assertIn("  3 | line3", out)
        self.assertIn("  4 | line4", out)
        self.assertNotIn("line2", out)
        self.assertIn("offset=5", out, "必须给出可执行的续读指令")

    def test_read_pages_cover_file_without_gaps(self):
        """100 行按 30 行一页读，拼起来必须与原文完全一致（不重不漏）。"""
        content = "\n".join(f"row-{i:03d}" for i in range(1, 101))
        self.tools["workspace_write_file"].invoke({"path": "a.txt", "content": content})

        seen: list[str] = []
        offset = 1
        for _ in range(10):
            out = self.tools["workspace_read_file"].invoke(
                {"path": "a.txt", "offset": offset, "limit": 30}
            )
            for line in out.split("\n"):
                if " | " in line and not line.startswith("..."):
                    seen.append(line.split(" | ", 1)[1])
            hit = re.search(r"offset=(\d+)", out)
            if not hit:
                break
            offset = int(hit.group(1))

        self.assertEqual(seen, [f"row-{i:03d}" for i in range(1, 101)])

    def test_read_offset_beyond_eof(self):
        self.tools["workspace_write_file"].invoke({"path": "a.txt", "content": "a\nb\n"})
        out = self.tools["workspace_read_file"].invoke({"path": "a.txt", "offset": 99})
        self.assertTrue(out.startswith("错误："), out)
        self.assertIn("共 2 行", out)

    def test_edit_tolerates_line_number_prefix(self):
        """模型从 read_file 输出里连行号一起复制时，edit 必须能兜住。

        这是加行号带来的必然风险：行号不是文件内容，带上就永远匹配不上，
        而报错只会说「未找到」，模型不知道毛病出在哪。
        """
        self.tools["workspace_write_file"].invoke(
            {"path": "a.py", "content": "def f():\n    return 1\n"}
        )
        out = self.tools["workspace_edit_file"].invoke(
            {
                "path": "a.py",
                "old_string": "  2 |     return 1",
                "new_string": "  2 |     return 2",
            }
        )
        self.assertIn("已修改", out)
        self.assertIn("return 2", (Path(self._tmp.name) / "a.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
