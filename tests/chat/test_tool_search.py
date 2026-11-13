"""工具检索（U11 P0-C）：分词、字段加权 BM25、destructive 降权、评测集回归。

## 这个文件的两类断言

- **行为断言**（分词 / 参数名 / 降权 / 平台加权）：单条逻辑对不对
- **评测集断言**（top-5 / top-1 命中率）：整体排序有没有退化

第二类更重要。把 `NAME_WEIGHT` 从 3 调到 2，大部分查询看起来还能用，
只有少数几条悄悄排错——这种退化只有回归指标能拦住。

调参时看 `scripts/eval_tool_search.py` 的逐条明细，不要靠这里猜。
"""

from __future__ import annotations

import unittest

from app.chat.tool_search import (
    ToolDoc,
    ToolIndex,
    build_docs,
    tokenize_tool_text,
)
from tests.chat.fixtures.tool_catalog import QUERIES, build_tools


class TokenizeTests(unittest.TestCase):
    def test_stopwords_are_dropped(self):
        """虚词必须丢：它们几乎出现在每条描述里，IDF 压不住，会主导排序。"""
        tokens = tokenize_tool_text("列出最近的 commit 记录")
        self.assertNotIn("的", tokens)
        self.assertNotIn("最近", tokens)
        self.assertIn("commit", tokens)

    def test_single_cjk_chars_are_dropped(self):
        """单字丢弃：jieba 会把「工单字段」切成 `工`/`单字`/`段`，单字全是噪音。"""
        tokens = tokenize_tool_text("上传文件到频道")
        self.assertNotIn("上", tokens)
        self.assertNotIn("到", tokens)

    def test_no_camel_split(self):
        """★ 不做驼峰拆分。

        第一版把 `GitHub` 拆成 `github`+`git`+`hub`，结果描述里写了 "GitHub"
        的工具白拿三份分，把 `github__search_code` 顶到了「在 GitHub 上开个工单」
        的第一名。
        """
        self.assertEqual(tokenize_tool_text("GitHub"), ["github"])
        self.assertNotIn("git", tokenize_tool_text("GitHub"))

    def test_plural_normalized(self):
        """工具名大量用复数（list_tables），查询给单数（表 → table）。"""
        self.assertEqual(tokenize_tool_text("tables"), tokenize_tool_text("table"))
        self.assertEqual(tokenize_tool_text("issues"), tokenize_tool_text("issue"))

    def test_plural_rule_is_conservative(self):
        """`status` / `analysis` 不该被砍成 `statu` / `analysi`。"""
        self.assertEqual(tokenize_tool_text("status"), ["status"])
        self.assertEqual(tokenize_tool_text("analysis"), ["analysis"])

    def test_tool_name_splits_on_underscore(self):
        self.assertEqual(
            tokenize_tool_text("github__create_issue"), ["github", "create", "issue"]
        )

    def test_synonyms_only_on_query_side(self):
        """文档侧不扩展：否则同义词表一改，所有已建索引的语义都会漂移。"""
        doc_tokens = tokenize_tool_text("创建 issue")
        query_tokens = tokenize_tool_text("创建 issue", expand_synonyms=True)
        self.assertNotIn("create", doc_tokens)
        self.assertIn("create", query_tokens)

    def test_single_char_synonym_still_applies(self):
        """★ 单字也要能查同义词。

        「建」是单字，会被长度过滤丢掉；但它的英文映射 `create` 正是需要的信号。
        第一版把同义词扩展放在过滤**之后**，注释写对了、代码顺序反了，
        导致「建」「删」「查」「表」的映射全部失效。
        """
        tokens = tokenize_tool_text("建一个页面", expand_synonyms=True)
        self.assertIn("create", tokens)
        self.assertNotIn("建", tokens)


class ParamIndexTests(unittest.TestCase):
    """参数名必须进索引：用户常说"按 owner 和 repo 查"，那是参数不是描述。"""

    def _index(self):
        docs = [
            ToolDoc(
                name="github__list_issues",
                description="列出仓库中的工单",
                params=["owner", "repo", "state"],
            ),
            ToolDoc(
                name="slack__send_message",
                description="向频道发送消息",
                params=["channel", "text"],
            ),
        ]
        return ToolIndex(docs)

    def test_param_name_matches(self):
        hits = [d.name for _, d in self._index().rank("按 repo 过滤")]
        self.assertIn("github__list_issues", hits)

    def test_param_does_not_behave_like_description(self):
        """参数命中权重高于描述，但不是万能的。"""
        hits = [d.name for _, d in self._index().rank("channel")]
        self.assertEqual(hits[0], "slack__send_message")


class DestructivePenaltyTests(unittest.TestCase):
    def _index(self):
        docs = [
            ToolDoc(name="sentry__list_issues", description="列出报错 issue"),
            ToolDoc(name="sentry__delete_issue", description="删除报错 issue"),
        ]
        return ToolIndex(docs)

    def test_destructive_is_pushed_down(self):
        """「找 issue」不该让 delete_issue 排第一——模型可能真的去调它。"""
        ranked = [d.name for _, d in self._index().rank("找 issue")]
        self.assertEqual(ranked[0], "sentry__list_issues")

    def test_destructive_allowed_when_intent_present(self):
        """用户明确说删，就必须能搜到。降权不是排除。"""
        ranked = [d.name for _, d in self._index().rank("删除 issue")]
        self.assertIn("sentry__delete_issue", ranked)


class ServerBoostTests(unittest.TestCase):
    def test_named_platform_wins(self):
        """说了 GitHub 就不该先给 Jira 的同名工具。"""
        docs = [
            ToolDoc(name="jira__create_issue", description="创建工单", server="jira"),
            ToolDoc(name="github__create_issue", description="创建工单", server="github"),
        ]
        ranked = [d.name for _, d in ToolIndex(docs).rank("在 GitHub 上创建工单")]
        self.assertEqual(ranked[0], "github__create_issue")


class EvalSetRegressionTests(unittest.TestCase):
    """评测集回归：防止排序整体退化。

    调参前先跑 `scripts/eval_tool_search.py` 看是哪几条会变，
    不要靠改这里的阈值让测试变绿。
    """

    @classmethod
    def setUpClass(cls):
        docs = build_docs(build_tools(), server_of=lambda n: n.rsplit("__", 1)[0])
        cls.index = ToolIndex(docs)

    def _rank(self, query: str, k: int = 5) -> list[str]:
        return [d.name for _, d in self.index.rank(query, k)]

    def test_top5_hit_rate(self):
        """top-5 是网关实际依赖的指标：模型会看到 5 个候选自己挑。"""
        hits = sum(
            1
            for q, expected, _ in QUERIES
            if set(self._rank(q)) & expected
        )
        rate = hits / len(QUERIES)
        self.assertGreaterEqual(
            rate,
            0.95,
            f"top-5 命中率跌到 {rate:.0%}（{hits}/{len(QUERIES)}），"
            "跑 scripts/eval_tool_search.py -v 看是哪几条",
        )

    def test_top1_accuracy(self):
        """top-1 是质量指标，不是硬约束。

        已知有 3 类会排错，都是 BM25「短文档占优」的固有特性：
        - 「提交一个 pull request」→ merge_pull_request 先于 create_pull_request
        - 「在 Notion 里建一个页面」→ get_page 先于 create_page
        - 「数据库里有哪些表」→ query_database 先于 list_tables
        这几条 top-5 都兜住了，模型能挑对，所以不为此继续调参（会过拟合评测集）。
        """
        hits = sum(
            1
            for q, expected, _ in QUERIES
            if (ranked := self._rank(q)) and ranked[0] in expected
        )
        rate = hits / len(QUERIES)
        self.assertGreaterEqual(
            rate,
            0.75,
            f"top-1 准确率跌到 {rate:.0%}（{hits}/{len(QUERIES)}）",
        )

    def test_empty_catalog_does_not_crash(self):
        self.assertEqual(ToolIndex([]).rank("随便"), [])


if __name__ == "__main__":
    unittest.main()
