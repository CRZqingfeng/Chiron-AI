"""Okapi BM25 打分（纯标准库，两处共用）。

## 为什么放在 `app/core`

原本只有会话附件检索一份实现（`app/chat/attachment_bm25_search.py`）。
U11 P0-C 给工具检索也加了 BM25，两套算法各自演化迟早会在参数和边界上分叉——
同一份"相关性"在两种检索里给出不同结果，排查时几乎不可能看出来。

`core` 是最底层，`app/chat` 与 `app/kb` 都能引用，所以放这里。

## 为什么只抽打分、不抽分词

分词需求不同：
- 附件是对**长文本**分词，jieba 效果好（`attachment_bm25_search._tokenize`）
- 工具名是 `github__create_issue` 这种标识符，要按 `__` / `_` / 驼峰切，
  jieba 会把它切成奇怪的片段（`tool_search.tokenize_tool_text`）

强行统一分词会让其中一边变差，所以只共享打分这一层。
"""

from __future__ import annotations

import math
from collections import Counter


def bm25_scores(
    corpus_tokens: list[list[str]],
    query_tokens: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """对每个文档算 BM25 分。

    :param corpus_tokens: 每个文档已分好的词列表
    :param query_tokens: 查询词列表
    :param k1: 词频饱和系数。1.2~2.0，越大越奖励重复出现
    :param b: 长度归一化强度。0=不归一，1=完全归一
    :return: 与 `corpus_tokens` 等长的分数列表；空语料返回空列表
    """
    n_docs = len(corpus_tokens)
    if n_docs == 0:
        return []
    doc_lens = [len(d) for d in corpus_tokens]
    avgdl = sum(doc_lens) / n_docs
    if avgdl <= 0:
        avgdl = 1.0
    df: dict[str, int] = {}
    for doc in corpus_tokens:
        for w in set(doc):
            df[w] = df.get(w, 0) + 1
    # ★ 用 `log(1 + ...)` 而不是教科书上的 `log((N - df + 0.5) / (df + 0.5))`。
    #
    # 教科书式在 `df > N/2` 时会变成**负数**：当一个词出现在半数以上文档里，
    # 整个文档得分可能为负。调用方一般用 `score > 0` 过滤候选，于是**候选全部消失**。
    # 实测踩到过：2 个工具的小语料里查「issue」，两个工具的 name 都含 issue，
    # `rank()` 直接返回空列表——工具搜索什么都没找到。
    #
    # Lucene 用的是 `log(1 + ...)`，恒正且保持单调。对只取 top-k 的调用方
    # （附件检索）排序结果完全一致，只是绝对分值不同。
    idf = {w: math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    scores: list[float] = []
    for idx, doc in enumerate(corpus_tokens):
        dl = doc_lens[idx]
        freq = Counter(doc)
        sc = 0.0
        for w in query_tokens:
            f = freq.get(w, 0)
            if f <= 0:
                continue
            denom = f + k1 * (1.0 - b + b * dl / avgdl)
            sc += idf.get(w, 0.0) * (f * (k1 + 1.0)) / denom
        scores.append(sc)
    return scores
