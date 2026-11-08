"""工作区文件操作原语：精确编辑的匹配语义与边界。

重点守两件最容易出事的事：

1. **CRLF 文件 + 模型给的 LF `old_string`**。模型从 `read_file` 看到的是文本，
   不会保留 `\\r`；字符级 `str.replace` 在这里必然匹配失败，而报错写着「未找到」，
   模型会以为文件被改了。本模块在归一化换行的文本上匹配，写回时再还原。
2. **多处匹配必须先报错**，绝不能默默改第一条——否则模型以为改了全部，
   用户以为只改了一处，两边都错。

用 LocalWorkspace 建临时目录跑，不启容器：ops 层与后端无关，
容器差异已在 test_docker 覆盖。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.workspace.local import LocalWorkspace
from app.workspace.ops import detect_newline, edit_text, glob_files, grep_files, is_binary


class EditTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))
        self.root = self.ws.root

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, content: str | bytes) -> Path:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        return target

    # --- 正常路径 ---------------------------------------------------------

    def test_unique_match_replaces_only_that_span(self):
        f = self._write("a.py", "def f():\n    return 1\n\nprint(f())\n")
        out = edit_text(self.root, "a.py", "    return 1", "    return 2")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["occurrences"], 1)
        self.assertEqual(out["first_line"], 2)
        self.assertEqual(f.read_text(encoding="utf-8"), "def f():\n    return 2\n\nprint(f())\n")

    def test_inline_fragment_is_not_line_bound(self):
        """只替换行内片段（不占整行）——Claude Code 的 Edit 也是子串语义。"""
        f = self._write("a.py", "x = compute(a, b)\n")
        out = edit_text(self.root, "a.py", "compute(a, b)", "compute(a, c)")
        self.assertTrue(out["ok"], out)
        self.assertEqual(f.read_text(encoding="utf-8"), "x = compute(a, c)\n")

    def test_chinese_content(self):
        f = self._write("notes.md", "# 记录\n\n第二行 TODO 要办\n第三行\n")
        out = edit_text(self.root, "notes.md", "第二行 TODO 要办", "第二行 已办")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["first_line"], 3)
        self.assertIn("第二行 已办", f.read_text(encoding="utf-8"))

    def test_context_rows_and_line_numbers(self):
        content = "".join(f"line{i}\n" for i in range(1, 11))
        self._write("a.txt", content)
        out = edit_text(self.root, "a.txt", "line5", "LINE5")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["total_lines"], 10)
        self.assertEqual([n for n, _ in out["before"]], [2, 3, 4])
        self.assertEqual([n for n, _ in out["after"]], [5, 6, 7])
        self.assertEqual(out["after"][0][1], "LINE5")

    # --- CRLF：本次改动的核心 ----------------------------------------------

    def test_crlf_file_accepts_lf_old_string(self):
        """模型给的片段行尾是 \\n，文件是 \\r\\n——必须能匹配，且写回仍是 CRLF。"""
        f = self._write("a.py", "def f():\r\n    return 1\r\n\r\nprint(f())\r\n")
        out = edit_text(self.root, "a.py", "    return 1\n", "    return 2\n")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["newline"], "CRLF")
        self.assertEqual(f.read_bytes(), b"def f():\r\n    return 2\r\n\r\nprint(f())\r\n")

    def test_lf_file_stays_lf(self):
        f = self._write("a.py", "a\nb\nc\n")
        out = edit_text(self.root, "a.py", "b\r\n", "B\r\n")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["newline"], "LF")
        self.assertEqual(f.read_bytes(), b"a\nB\nc\n")

    def test_mixed_newlines_reports_note(self):
        f = self._write("a.py", "a\r\nb\nc\r\nd\n")
        out = edit_text(self.root, "a.py", "b", "B")
        self.assertTrue(out["ok"], out)
        self.assertIn("混用换行", out["note"])
        # 混用时统一成主导行尾符，必须如实告知（已在上一条断言里）
        self.assertIn(b"\r\n", f.read_bytes())

    # --- 多处匹配：必须先报错 ----------------------------------------------

    def test_ambiguous_match_fails_without_touching_file(self):
        original = "foo\nbar\nfoo\n"
        f = self._write("a.txt", original)
        out = edit_text(self.root, "a.txt", "foo", "baz")
        self.assertFalse(out["ok"])
        self.assertEqual(out["code"], "ambiguous")
        self.assertEqual(out["occurrences"], 2)
        self.assertEqual(out["match_lines"], [1, 3])
        self.assertEqual(f.read_text(encoding="utf-8"), original, "报错时不能改动文件")

    def test_replace_all(self):
        f = self._write("a.txt", "foo\nbar\nfoo\n")
        out = edit_text(self.root, "a.txt", "foo", "baz", replace_all=True)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["occurrences"], 2)
        self.assertEqual(f.read_text(encoding="utf-8"), "baz\nbar\nbaz\n")

    def test_not_found(self):
        self._write("a.txt", "hello\n")
        out = edit_text(self.root, "a.txt", "missing", "x")
        self.assertFalse(out["ok"])
        self.assertEqual(out["code"], "not_found")
        self.assertEqual(out["total_lines"], 1)

    # --- 拒绝面 -----------------------------------------------------------

    def test_empty_old_string_rejected(self):
        self._write("a.txt", "hello\n")
        self.assertEqual(edit_text(self.root, "a.txt", "", "x")["code"], "empty_old")

    def test_same_old_and_new_rejected(self):
        self._write("a.txt", "hello\n")
        self.assertEqual(edit_text(self.root, "a.txt", "hello", "hello")["code"], "same")

    def test_whitespace_only_replace_all_rejected(self):
        self._write("a.txt", "a\n\nb\n\nc\n")
        self.assertEqual(
            edit_text(self.root, "a.txt", "\n\n", "\n", replace_all=True)["code"],
            "unsafe_replace_all",
        )

    def test_binary_file_rejected(self):
        self._write("blob.bin", b"\x00\x01\x02binary")
        self.assertEqual(edit_text(self.root, "blob.bin", "a", "b")["code"], "not_text")

    def test_non_utf8_file_rejected(self):
        """严格解码：用 replace 兜底会把 U+FFFD 写回磁盘，原文就回不去了。"""
        self._write("gbk.txt", "中文内容".encode("gbk"))
        self.assertEqual(edit_text(self.root, "gbk.txt", "中文", "英文")["code"], "not_text")

    def test_missing_file_and_directory(self):
        self.assertEqual(edit_text(self.root, "nope.txt", "a", "b")["code"], "no_file")
        (self.root / "dir").mkdir()
        self.assertEqual(edit_text(self.root, "dir", "a", "b")["code"], "is_dir")

    def test_path_escape_rejected(self):
        self.assertEqual(edit_text(self.root, "../outside.txt", "a", "b")["code"], "bad_path")
        self.assertEqual(edit_text(self.root, str(self.root / "abs.txt"), "a", "b")["code"], "bad_path")

    def test_oversized_file_rejected(self):
        big = "a" * (2 * 1024 * 1024 + 10)
        self._write("big.txt", big)
        self.assertEqual(edit_text(self.root, "big.txt", "a", "b")["code"], "too_large")


class OpsHelperTests(unittest.TestCase):
    def test_detect_newline(self):
        self.assertEqual(detect_newline("a\r\nb\r\n"), ("\r\n", False))
        self.assertEqual(detect_newline("a\nb\n"), ("\n", False))
        self.assertEqual(detect_newline("a\r\nb\nc\r\n"), ("\r\n", True))
        self.assertEqual(detect_newline("a\r\nb\nc\n"), ("\n", True))

    def test_is_binary(self):
        self.assertTrue(is_binary(b"ab\x00cd"))
        self.assertFalse(is_binary("纯文本".encode()))


class GrepTests(unittest.TestCase):
    """内容检索：语义、跳过规则、截断与提示。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))
        self.root = self.ws.root
        self._write("a.py", "def compute():\n    return foo\n")
        self._write("b.py", "x = foo(1)\n")
        self._write("notes.md", "TODO: foo 要改\n")
        self._write("sub/deep/d.py", "# foo 在深目录\n")
        # 这两处必须被跳过，否则「搜不到 = 没有」的判断就是错的
        self._write(".git/config", "foo\n")
        self._write("node_modules/m.js", "foo\n")
        self._write("blob.bin", b"\x00foo")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, content: str | bytes) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)

    def test_literal_content_hits_with_line_numbers(self):
        out = grep_files(self.root, "foo")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["total_matches"], 4)
        self.assertEqual(len(out["hits"]), 4)
        self.assertIn(("a.py", 2), [(h["path"], h["line"]) for h in out["hits"]])
        self.assertIn(("sub/deep/d.py", 1), [(h["path"], h["line"]) for h in out["hits"]])

    def test_skips_git_node_modules_and_binary(self):
        out = grep_files(self.root, "foo")
        paths = [h["path"] for h in out["hits"]]
        self.assertNotIn(".git/config", paths)
        self.assertNotIn("node_modules/m.js", paths)
        self.assertEqual(out["skipped_binary"], 1)

    def test_literal_matches_regex_metachars_verbatim(self):
        """`a(b` 按字面能搜到——默认不是正则，这是刻意的。"""
        self._write("weird.txt", "call a(b here\n")
        out = grep_files(self.root, "a(b")
        self.assertTrue(out["ok"], out)
        self.assertEqual([h["path"] for h in out["hits"]], ["weird.txt"])

    def test_no_hit_with_metachars_sets_regex_hint(self):
        out = grep_files(self.root, "foo|bar")
        self.assertEqual(out["total_matches"], 0)
        self.assertTrue(out["regex_hint"])

    def test_regex_mode_works(self):
        out = grep_files(self.root, "foo|TODO", regex=True)
        self.assertTrue(out["ok"], out)
        self.assertGreaterEqual(out["total_matches"], 4)

    def test_bad_regex_reports_error(self):
        out = grep_files(self.root, "a(b", regex=True)
        self.assertFalse(out["ok"])
        self.assertEqual(out["code"], "bad_regex")

    def test_glob_filter(self):
        """`glob` 是**文件过滤器**，`*.py` 匹配所有层级的 .py（不只当前目录）。"""
        out = grep_files(self.root, "foo", glob="*.py")
        self.assertEqual({h["path"] for h in out["hits"]}, {"a.py", "b.py", "sub/deep/d.py"})
        # 带 `/` 的模式按相对路径匹配
        nested = grep_files(self.root, "foo", glob="sub/**/*.py")
        self.assertEqual({h["path"] for h in nested["hits"]}, {"sub/deep/d.py"})

    def test_three_modes(self):
        content = grep_files(self.root, "foo", mode="content")
        files = grep_files(self.root, "foo", mode="files_with_matches")
        counts = grep_files(self.root, "foo", mode="count")

        self.assertEqual(len(content["hits"]), 4)
        self.assertEqual(sorted(files["files"]), ["a.py", "b.py", "notes.md", "sub/deep/d.py"])
        self.assertEqual(dict(counts["counts"])["a.py"], 1)

    def test_head_limit_truncates(self):
        out = grep_files(self.root, "foo", head_limit=1)
        self.assertEqual(len(out["hits"]), 1)
        self.assertTrue(out["truncated"])

    def test_path_scope_limits_search(self):
        out = grep_files(self.root, "foo", path="sub")
        self.assertEqual([h["path"] for h in out["hits"]], ["sub/deep/d.py"])

    def test_path_escape_rejected(self):
        out = grep_files(self.root, "foo", path="../")
        self.assertFalse(out["ok"])
        self.assertEqual(out["code"], "bad_path")

    def test_long_line_clipped_around_match(self):
        """匹配点在行尾时不能被切掉——这是「居中开窗」而不是 `line[:300]` 的理由。"""
        self._write("long.txt", "x" * 500 + "NEEDLE" + "y" * 500 + "\n")
        out = grep_files(self.root, "NEEDLE")
        text = out["hits"][0]["text"]
        self.assertIn("NEEDLE", text)
        self.assertLessEqual(len(text), 310)


class GlobTests(unittest.TestCase):
    """文件名匹配：递归、剪枝、限额、模式校验。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))
        self.root = self.ws.root
        for name in ("a.py", "b.py", "readme.md", "src/util/c.py", ".git/hook.py", "node_modules/x/y.py"):
            self._write(name, "x\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, content: str) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def test_recursive_double_star(self):
        out = glob_files(self.root, "**/*.py")
        paths = [it["path"] for it in out["items"]]
        self.assertIn("a.py", paths)
        self.assertIn("src/util/c.py", paths)
        # .git 与 node_modules 里的不能被列出来
        self.assertNotIn(".git/hook.py", paths)
        self.assertNotIn("node_modules/x/y.py", paths)

    def test_single_level_pattern(self):
        out = glob_files(self.root, "*.md")
        self.assertEqual([it["path"] for it in out["items"]], ["readme.md"])

    def test_no_match(self):
        out = glob_files(self.root, "*.rs")
        self.assertTrue(out["ok"])
        self.assertEqual(out["items"], [])

    def test_deterministic_order(self):
        first = [it["path"] for it in glob_files(self.root, "**/*.py")["items"]]
        second = [it["path"] for it in glob_files(self.root, "**/*.py")["items"]]
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))

    def test_limit_truncates(self):
        out = glob_files(self.root, "**/*.py", limit=2)
        self.assertEqual(len(out["items"]), 2)
        self.assertTrue(out["truncated"])

    def test_pattern_escape_rejected(self):
        self.assertEqual(glob_files(self.root, "../*.py")["code"], "bad_pattern")
        self.assertEqual(glob_files(self.root, "/etc/*")["code"], "bad_pattern")

    def test_path_escape_rejected(self):
        self.assertEqual(glob_files(self.root, "*.py", path="../")["code"], "bad_path")


if __name__ == "__main__":
    unittest.main()
