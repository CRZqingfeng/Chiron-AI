# -*- coding: utf-8 -*-
"""Markdown HTML 噪音剥离的单测。

背景：FastAPI 等项目的 README 开头有大段 logo / badge / sponsor HTML，
入库后被切成独立块，既召回不到答案又挤占 top_k 名额
（实测第三组 #3/#4 的 5 个注入块里有 3 个是这种）。

这类文本清洗最容易误伤，所以把边界情况都锁住：代码块要保护、泛型不能当标签删、
徽章行要清掉、纯文本不能被改动。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from app.kb.multimodal_document_loader import _load_text_pages, has_substance, strip_html_noise
from app.settings import settings


class StripHtmlNoiseTests(unittest.TestCase):
    def test_decorative_html_block_removed(self):
        src = (
            '<p align="center">\n'
            '  <a href="https://fastapi.tiangolo.com"><img src="logo.png" alt="FastAPI"></a>\n'
            "</p>\n\n"
            "## 正文标题\n"
            "这是正文内容。"
        )
        out = strip_html_noise(src)
        self.assertNotIn("<p", out)
        self.assertNotIn("<img", out)
        self.assertNotIn("logo.png", out)
        self.assertIn("## 正文标题", out)
        self.assertIn("这是正文内容", out)

    def test_fenced_code_block_is_protected(self):
        """讲 HTML/JSX 的文档，代码块里的标签是正文，不能剥。"""
        src = '示例：\n```html\n<div class="a">hi</div>\n```\n后面一句话'
        out = strip_html_noise(src)
        self.assertIn('<div class="a">hi</div>', out)
        self.assertIn("后面一句话", out)

    def test_generic_type_not_treated_as_tag(self):
        """List<String> / Map<K,V> 不是 HTML 标签，不能被删。"""
        src = "定义 List<String> 和 Map<K,V> 两种类型"
        self.assertEqual(strip_html_noise(src), src)

    def test_markdown_badge_line_removed(self):
        src = (
            "[![Build Status](https://github.com/x/y/workflows/CI/badge.svg)]"
            "(https://github.com/x/y/actions)\n"
            "这里是正文。"
        )
        out = strip_html_noise(src)
        self.assertNotIn("badge.svg", out)
        self.assertIn("这里是正文", out)

    def test_inline_tag_keeps_inner_text(self):
        """<a> 这类内联标签要保留文字内容，不能整段丢。"""
        src = "详见 <a href='https://x.dev'>官方文档</a> 的说明"
        out = strip_html_noise(src)
        self.assertIn("官方文档", out)
        self.assertNotIn("<a", out)

    def test_plain_text_unchanged(self):
        src = "普通段落，没有任何标签。\n第二行"
        self.assertEqual(strip_html_noise(src), src)

    def test_empty_input(self):
        self.assertEqual(strip_html_noise(""), "")


class HasSubstanceTests(unittest.TestCase):
    def test_blank_and_punctuation_only(self):
        self.assertFalse(has_substance(""))
        self.assertFalse(has_substance("   \n\n --- "))     # 只剩标点空白
        self.assertFalse(has_substance("[](https://x)"))    # 纯链接残渣

    def test_real_text_passes(self):
        self.assertTrue(has_substance("这是一个有内容的段落"))
        self.assertTrue(has_substance("hello world"))

    def test_threshold_respected(self):
        self.assertFalse(has_substance("abc", min_chars=10))
        self.assertTrue(has_substance("abcdefghij", min_chars=10))


class LoadTextPagesTests(unittest.TestCase):
    def _load(self, content: str):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name
        try:
            return _load_text_pages(path)
        finally:
            os.unlink(path)

    def test_html_noise_stripped_on_load(self):
        with patch.object(settings, "KB_STRIP_HTML_ENABLED", True):
            pages = self._load('<p align="center"><img src="logo.png"></p>\n\n# 标题\n正文内容在这里。')
        self.assertEqual(len(pages), 1)
        self.assertNotIn("<img", pages[0].page_content)
        self.assertIn("# 标题", pages[0].page_content)

    def test_switch_off_keeps_raw(self):
        with patch.object(settings, "KB_STRIP_HTML_ENABLED", False):
            pages = self._load('<p align="center"><img src="logo.png"></p>\n\n正文')
        self.assertIn("<img", pages[0].page_content)


if __name__ == "__main__":
    unittest.main()
