"""BM25 核心（`app/core/bm25.py`）。

这个模块被两处共用（附件检索 + MCP 工具检索），所以它的边界必须单独守住——
两边各写一套测试的话，出问题时不知道该信哪边的断言。

重点是 **IDF 恒正**：教科书公式在 `df > N/2` 时变负，调用方一旦用 `score > 0`
过滤候选，就会出现「明明有匹配，结果却是空列表」。
"""

from __future__ import annotations

import unittest

from app.core.bm25 import bm25_scores


class Bm25ScoreTests(unittest.TestCase):
    def test_ranking_prefers_more_matches(self):
        # 三篇等长：BM25 的长度归一化会让短文档占便宜，等长才比得出「命中更多」
        corpus = [
            ["create", "issue"],
            ["list", "ticket"],
            ["delete", "issue"],
        ]
        scores = bm25_scores(corpus, ["create", "issue"])
        self.assertGreater(scores[0], scores[1], "命中一个词的应高于零命中")
        self.assertEqual(max(range(len(scores)), key=lambda i: scores[i]), 0)

    def test_shorter_doc_wins_on_equal_hits(self):
        """长度归一化是 BM25 的固有行为，不是 bug——记录下来免得后人以为是错的。"""
        corpus = [["create", "issue"], ["create", "issue", "github", "extra"]]
        scores = bm25_scores(corpus, ["create", "issue"])
        self.assertGreater(scores[0], scores[1])

    def test_no_match_is_zero(self):
        scores = bm25_scores([["a"], ["b"]], ["zzz"])
        self.assertEqual(scores, [0.0, 0.0])

    def test_empty_corpus(self):
        self.assertEqual(bm25_scores([], ["a"]), [])

    def test_empty_query(self):
        self.assertEqual(bm25_scores([["a"], ["b"]], []), [0.0, 0.0])

    def test_idf_stays_positive_when_term_appears_everywhere(self):
        """★ 每个文档都含这个词时，分数必须仍然是正的。

        教科书式 `log((N - df + 0.5) / (df + 0.5))` 在这里给出 log(0.2) < 0，
        整个文档得分为负；调用方用 `score > 0` 过滤候选时结果直接变成空列表。
        实测踩过：2 个工具的小语料里查 "issue"，`rank()` 返回空。
        """
        corpus = [["issue", "create"], ["issue", "list"]]
        scores = bm25_scores(corpus, ["issue"])
        self.assertTrue(all(s > 0 for s in scores), f"IDF 变负了：{scores}")

    def test_scores_are_finite(self):
        corpus = [["a"], ["a", "b"], ["b", "c"]]
        for s in bm25_scores(corpus, ["a", "b", "c"]):
            self.assertAlmostEqual(s, s, msg="出现 NaN")


if __name__ == "__main__":
    unittest.main()
