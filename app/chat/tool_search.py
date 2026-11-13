"""工具检索：字段加权 BM25 + destructive 降权。

## 为什么不能只按词面命中排序

P0-B 的最小实现是「词出现在名字里 +3、出现在描述里 +1」，够跑通闭环，
但有两个明显问题：

1. **没有 IDF**：`search`、`list` 这种词几乎每个工具描述里都有，命中它们不代表相关；
   而 `merge request`、`webhook` 这种稀有词才是真正的判别信号。BM25 的 IDF 正好治这个。
2. **不看参数名**：用户说「按 owner 和 repo 查」，`owner`/`repo` 往往只出现在
   **参数**里，不出现在描述里。只索引描述会漏。

## 三个字段分开算，而不是拼成一段

`github__create_issue` 这个名字已经是最强的信号（BM25 里字段边界明确），
拼进描述里会被长描述稀释。所以按字段分别打分再加权：

    score = 3.0 × bm25(name) + 1.0 × bm25(description) + 2.0 × bm25(params)

参数名权重低于名字、高于描述：它是"能推断出用途"的强信号，但不如名字直接。

## destructive 为什么要降权

`delete_repository`、`archive_project` 的描述里常写着 "用于删除……"，
在「找创建 issue 的工具」这种查询里会因 `issue` 命中而被带上来。
它们排进候选的代价不是多点一次，而是**模型可能真的去调它**。
所以：工具本身是破坏性的、而查询没有表现出破坏意图时，分数乘 `DESTRUCTIVE_PENALTY`。

**不是直接排除**：用户明确说"删掉这个 issue"时必须能搜到。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.core.bm25 import bm25_scores

# 字段权重（实测调过：名字权重再低会让 create/list 难以区分，再高会压掉描述里的关键信号）
NAME_WEIGHT = 3.0
DESC_WEIGHT = 1.0
PARAM_WEIGHT = 2.0

# 破坏性动作词：出现在工具名/描述里即视为"该工具会改变状态"
_DESTRUCTIVE_TOKENS = frozenset(
    {
        "delete",
        "remove",
        "destroy",
        "drop",
        "purge",
        "truncate",
        "archive",
        "close",
        "reset",
        "revoke",
        "cancel",
        "disable",
        "deactivate",
        "uninstall",
        "wipe",
        "kill",
    }
)

# 查询里出现这些词，说明用户确实想要破坏性操作，此时不降权
_INTENT_TOKENS = frozenset(
    {
        "delete",
        "remove",
        "destroy",
        "drop",
        "purge",
        "truncate",
        "archive",
        "close",
        "reset",
        "revoke",
        "cancel",
        "disable",
        "deactivate",
        "uninstall",
        "删除",
        "删掉",
        "移除",
        "清除",
        "销毁",
        "归档",
        "关闭",
        "作废",
        "撤销",
        "注销",
        "下线",
        "禁用",
    }
)

# 同义词扩展词的权重。低于原词是因为「中文意图 → 英文关键词」只是一次猜测：
# 「仓库」映射到 repo + repository 后，若与原词等权，含 repository 的工具会
# 白拿两份名次分（实测把 `github__fork_repository` 顶到了「读 README」的第一名）。
SYNONYM_WEIGHT = 0.5

DESTRUCTIVE_PENALTY = 0.3

# 用户明确提到某个平台时（"在 GitHub 上开个工单"），该平台的工具加权。
# 这是最没有歧义的一类信号：说了 GitHub 就不该先给 Jira 的 create_issue。
# 不加权的实测结果是 jira__create_issue 排第一——因为它的描述更短，
# 在 BM25 的长度归一化里占了便宜。
SERVER_BOOST = 1.6

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 中文虚词：它们在几乎每条描述/参数说明里都出现，BM25 的 IDF 压不住
# （54 个文档里"的"出现在 40 个，仍然不是 0 分），实测会主导排序。
# 典型翻车：「列出最近的 commit 记录」里 `的` 让 `jira__update_issue` 排到了
# `github__list_commits` 前面。
_STOPWORDS_CN = frozenset(
    {
        "的", "了", "在", "上", "中", "里", "和", "与", "及", "把", "被", "给",
        "对", "为", "从", "到", "用", "请", "帮", "我", "你", "他", "她", "它",
        "这", "那", "一个", "一下", "这个", "那个", "哪些", "哪里", "什么",
        "怎么", "如何", "是否", "可以", "需要", "最近", "然后", "以及", "一些",
        "帮我", "看看", "能不能", "有没有", "相关", "东西",
    }
)

_STOPWORDS_EN = frozenset({"a", "an", "the", "to", "of", "and", "or", "for", "in", "on", "with", "is", "it"})

# 中文意图 → 工具名里的英文动词/名词。
#
# ## 为什么这条是决定性的
#
# 用户用中文提问（"创建一个 issue"），而 MCP 工具名几乎都是英文
# （`github__create_issue`）。BM25 是词面匹配，"创建" 和 `create` **零重合**。
# 实测不扩同义词时，「创建一个 issue」的 top-1 是 `linear__get_issue`——
# `issue` 命中但 `create` 完全没参与，排序就变成了「谁描述里 issue 多谁赢」。
#
# ## 为什么只在查询侧扩展
#
# 文档侧保持不变，索引才稳定（改同义词表不会让已有工具的索引语义漂移）。
# 查询侧把中文意图"翻译"成英文关键词，是这一层唯一该做的语义归一。
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "创建": ("create",),
    "新建": ("create",),
    "建立": ("create",),
    "建": ("create",),
    "添加": ("add", "create"),
    "加": ("add", "comment"),
    "删除": ("delete", "remove"),
    "删掉": ("delete",),
    "删": ("delete",),
    "移除": ("remove",),
    "清除": ("delete", "purge"),
    "更新": ("update",),
    "修改": ("update",),
    "改": ("update",),
    "获取": ("get",),
    "读取": ("get", "read"),
    "读": ("get", "read"),
    "查看": ("get", "list"),
    "看": ("get", "list"),
    "列出": ("list",),
    "列表": ("list",),
    "搜索": ("search",),
    "搜": ("search",),
    "查找": ("search", "get"),
    "查": ("search", "query", "get"),
    "查询": ("query", "search"),
    "合并": ("merge",),
    "关闭": ("close",),
    "归档": ("archive",),
    "解决": ("resolve",),
    "标记": ("update", "resolve"),
    "指派": ("assign",),
    "分配": ("assign",),
    "邀请": ("invite",),
    "上传": ("upload",),
    "下载": ("download",),
    "发送": ("send",),
    "发": ("send",),
    "执行": ("query", "execute", "run"),
    "跑": ("query", "run"),
    "工单": ("issue",),
    "问题": ("issue",),
    "报错": ("issue", "error"),
    "错误": ("error", "issue"),
    "仓库": ("repo", "repository"),
    "项目": ("project",),
    "分支": ("branch",),
    "提交": ("commit", "create"),
    "记录": ("commit", "log"),
    "页面": ("page",),
    "频道": ("channel",),
    "消息": ("message",),
    "聊": ("get", "history"),
    "历史": ("history",),
    "评论": ("comment",),
    "文件": ("file",),
    "数据库": ("database", "db"),
    "表": ("table",),
    "结构": ("schema", "describe"),
    "代码": ("code",),
    "函数": ("code", "function"),
}


# 扩展词里**不允许参与 name 字段**的那部分。
#
# 判据是「这个词在工具名里是**稳定的意图信号**，还是**恰好是该工具的对象**」：
# - `table` / `issue` / `keyword` 这类，工具名里有它说明用途相关（`list_tables`）
# - `repository` / `repo` 这类不行：`github__fork_repository` 的名字里也有
#   `repository`，会让「读 README」把它顶到 `get_file_contents` 前面（实测）
#
# 一开始写的是白名单（只放动词），结果 `table` 被挡住，「数据库里有哪些表」
# 的第一名变成了 `notion__create_page`（它的参数名里含"数据库"）。
# 黑名单更贴合实际：需要保护的只是那几类"对象即名字"的词。
_NAME_BLOCKED = frozenset(
    {"repository", "repo", "project", "workspace", "organisation", "organization"}
)


def _singular(word: str) -> str:
    """英文单复数归一。

    MCP 工具名大量用复数（`list_issues` / `list_tables` / `list_pull_requests`），
    而用户和同义词表给的是单数。不做归一时 `table` 永远匹配不上 `tables`——
    实测「数据库里有哪些表」的第一名是 `notion__create_page`（它的参数名里有"数据库"）。

    只做最保守的规则：去掉词尾单个 `s`，跳过 `ss`/`us`/`is` 结尾
    （`status`/`analysis` 这类不该被砍）。误伤的词两边一致，不影响匹配。
    """
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _cjk_tokens(text: str) -> list[str]:
    """中文分词：jieba 优先，缺失时退化为 bigram。

    退化路径不是"没装就少个功能"——bigram 对「创建 issue」这类短查询已经够用，
    只是长句会引入一些噪音。
    """
    if not text:
        return []
    try:
        import jieba

        return [t.strip() for t in jieba.lcut(text) if t and t.strip() and not t.isspace()]
    except Exception:  # noqa: BLE001 - 分词器故障不能阻断搜索
        return [text[i : i + 2] for i in range(max(0, len(text) - 1))]


def tokenize_tool_text(text: str, *, expand_synonyms: bool = False) -> list[str]:
    """工具名/描述/参数名 的分词。

    :param expand_synonyms: 是否做中文→英文同义词扩展。**只在查询侧开**，
        文档侧保持原文，否则同义词表一改，所有已建索引的语义都会漂移。

    与附件检索的分词**刻意不同**：工具名是 `github__create_issue` 这种标识符，
    jieba 会把它切成奇怪片段，所以先按字母数字切成英文词，中文段才交给 jieba。

    ## 四条必须做的清理（每一条都是实测踩出来的）

    1. **不做驼峰拆分，只做整词小写**：`GitHub` → `github`，正好对上工具名里的
       `github`。第一版按驼峰拆成 `github`+`git`+`hub` 三份，结果是**描述里写了
       "GitHub" 的工具白拿三份分**——实测把 `github__search_code` 顶到了
       「在 GitHub 上开一个工单」的第一名，而它和"开"毫无关系。
    2. **英文单复数归一**：见 `_singular`。
    3. **中文单字丢弃**：`开`、`上`、`工` 这类单字在长描述里到处出现。
       jieba 还会把「工单字段」切成 `工`/`单字`/`段`，单字参与打分会把
       完全无关的工具顶上来。
    4. **虚词丢弃**：见 `_STOPWORDS_CN` 的注释。
    """
    s = str(text or "")
    if not s:
        return []

    out: list[str] = [
        _singular(w.lower())
        for w in re.findall(r"[A-Za-z0-9]+", s)
        if w.lower() not in _STOPWORDS_EN
    ]

    if _CJK_RE.search(s):
        cjk_text = "".join(re.findall(r"[\u4e00-\u9fff]+", s))
        for raw_token in _cjk_tokens(cjk_text):
            token = raw_token.strip()
            if not token:
                continue
            # ⚠️ 同义词必须在这里查，不能放到循环之后——「建」「删」「查」「表」
            # 这些**单字**马上会被长度过滤丢掉，丢完就再也查不到映射了
            # （第一版就是这么错的：注释写对了，代码顺序反了）。
            if expand_synonyms:
                out.extend(_singular(x) for x in _SYNONYMS.get(token, ()))
            if len(token) >= 2 and token not in _STOPWORDS_CN:
                out.append(token)

    return out


def _schema_param_tokens(args_schema: Any) -> list[str]:
    """从工具的 args_schema 里取「参数名 + 参数说明」。

    参数说明值得索引：`owner: 仓库所有者` 这种能让「仓库」这个中文词命中。
    """
    schema = args_schema if isinstance(args_schema, dict) else {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        return []
    out: list[str] = []
    for key, spec in props.items():
        out.append(str(key))
        if isinstance(spec, dict):
            desc = spec.get("description")
            if desc:
                out.append(str(desc))
    return out


@dataclass
class ToolDoc:
    """检索语料里的一条工具记录。"""

    name: str
    description: str = ""
    server: str = ""
    params: list[str] = field(default_factory=list)

    @property
    def is_destructive(self) -> bool:
        toks = set(tokenize_tool_text(f"{self.name} {self.description}"))
        return bool(toks & _DESTRUCTIVE_TOKENS)


class ToolIndex:
    """一次构建、多次查询。构建成本 O(工具数)，每次查询 O(工具数)。

    不做增量更新：工具目录每轮对话重建一次（MCP 列表本来就每轮重新加载），
    缓存反而会引入「目录过期」这类难排查的问题。
    """

    def __init__(self, docs: Iterable[ToolDoc]) -> None:
        self.docs: list[ToolDoc] = [d for d in docs if d.name]
        self._name_tokens = [tokenize_tool_text(d.name) for d in self.docs]
        self._desc_tokens = [tokenize_tool_text(d.description) for d in self.docs]
        self._param_tokens = [tokenize_tool_text(" ".join(d.params)) for d in self.docs]
        self._server_tokens = [set(tokenize_tool_text(d.server)) for d in self.docs]

    def __len__(self) -> int:
        return len(self.docs)

    def _field_scores(self, tokens: list[str]) -> tuple[list[float], list[float], list[float]]:
        if not tokens:
            zeros = [0.0] * len(self.docs)
            return zeros, zeros, zeros
        return (
            bm25_scores(self._name_tokens, tokens),
            bm25_scores(self._desc_tokens, tokens),
            bm25_scores(self._param_tokens, tokens),
        )

    def rank(self, query: str, limit: int = 8) -> list[tuple[float, ToolDoc]]:
        """返回按相关性降序的 `(score, doc)`，只保留分数 > 0 的。

        原词与同义词扩展词**分开算分**，后者乘 `SYNONYM_WEIGHT`：
        合并成一个 token 列表会让映射出多个词的中文（仓库 → repo/repository）
        白拿双份分数。
        """
        raw = tokenize_tool_text(query)
        expanded = tokenize_tool_text(query, expand_synonyms=True)
        if not expanded or not self.docs:
            return []
        raw_set = set(raw)
        extra = [t for t in expanded if t not in raw_set]

        n_name, n_desc, n_param = self._field_scores(raw)
        # 扩展词能否参与 name 由 `_NAME_BLOCKED` 决定（见该常量的注释）。
        e_name, e_desc, e_param = self._field_scores([t for t in extra if t not in _NAME_BLOCKED])

        raw_set = set(raw)
        wants_destructive = bool(set(expanded) & _INTENT_TOKENS)
        scored: list[tuple[float, ToolDoc]] = []
        for i, doc in enumerate(self.docs):
            score = (
                NAME_WEIGHT * (n_name[i] + SYNONYM_WEIGHT * e_name[i])
                + DESC_WEIGHT * (n_desc[i] + SYNONYM_WEIGHT * e_desc[i])
                + PARAM_WEIGHT * (n_param[i] + SYNONYM_WEIGHT * e_param[i])
            )
            if score <= 0:
                continue
            if self._server_tokens[i] & raw_set:
                score *= SERVER_BOOST
            if doc.is_destructive and not wants_destructive:
                score *= DESTRUCTIVE_PENALTY
            scored.append((score, doc))

        scored.sort(key=lambda x: (-x[0], x[1].name))
        return scored[: max(1, int(limit))]


def build_docs(tools: Iterable[Any], *, server_of: Any = None) -> list[ToolDoc]:
    """从 LangChain 工具对象构造检索语料。

    :param server_of: 由名字反解服务名的函数。放在调用方传，是因为
        `{server}__{tool}` 的拼法是 MCP 层的事，检索层不该知道这个约定。
    """
    docs: list[ToolDoc] = []
    for t in tools or ():
        name = str(getattr(t, "name", "") or "")
        if not name:
            continue
        docs.append(
            ToolDoc(
                name=name,
                description=str(getattr(t, "description", "") or ""),
                server=str(server_of(name)) if server_of else "",
                params=_schema_param_tokens(getattr(t, "args_schema", None)),
            )
        )
    return docs
