"""工具检索的评测集：54 个真实感 MCP 工具 + 20 条标注查询。

## 为什么必须有这个文件

调 BM25 的字段权重、`k1`/`b`、destructive 惩罚系数，如果不看真实查询的表现，
就是拍脑袋。更糟的是**退化不可见**：把 `NAME_WEIGHT` 从 3 调到 1，
大部分查询看起来还能用，只有少数几条悄悄排错了——这种问题只会被用户撞见。

## 数据集怎么来的

工具名与参数照抄各平台 MCP server 的真实形态（`github__create_issue` 这种
`{server}__{tool}` 命名正是 U11 P0-A 定的），描述用中文——这是本项目用户的实际形态。

`QUERIES` 里每条标注**所有可接受的答案**：用户说「创建 issue」时，
GitHub / Linear / Jira 的实现都算对，只要别返回 `archive_project`。

## 怎么用

- `tests/chat/test_tool_search.py` 断言 top-5 命中率（防止排序退化）
- `scripts/eval_tool_search.py` 打印每条明细（调参时看这个）
"""

from __future__ import annotations

from typing import Any

# (工具名, 描述, {参数名: 参数说明})
CATALOG: list[tuple[str, str, dict[str, str]]] = [
    # ---- GitHub ----
    ("github__create_issue", "创建一个新的 issue（工单）", {"owner": "仓库所有者", "repo": "仓库名", "title": "标题", "body": "正文"}),
    ("github__list_issues", "列出仓库中的 issue，支持按状态过滤", {"owner": "仓库所有者", "repo": "仓库名", "state": "状态：open/closed"}),
    ("github__get_issue", "获取单个 issue 的详细信息", {"owner": "仓库所有者", "repo": "仓库名", "issue_number": "工单编号"}),
    ("github__update_issue", "更新 issue 的标题、正文或标签", {"owner": "仓库所有者", "repo": "仓库名", "issue_number": "工单编号"}),
    ("github__create_pull_request", "创建一个 pull request", {"owner": "仓库所有者", "repo": "仓库名", "head": "源分支", "base": "目标分支", "title": "标题"}),
    ("github__merge_pull_request", "合并一个 pull request", {"owner": "仓库所有者", "repo": "仓库名", "pull_number": "PR 编号"}),
    ("github__list_pull_requests", "列出仓库的 pull request", {"owner": "仓库所有者", "repo": "仓库名"}),
    ("github__get_file_contents", "读取仓库中某个文件的内容（如 README、源码文件）", {"owner": "仓库所有者", "repo": "仓库名", "path": "文件路径"}),
    ("github__create_or_update_file", "新建或覆盖仓库中的文件并提交", {"owner": "仓库所有者", "repo": "仓库名", "path": "文件路径", "content": "文件内容", "message": "提交说明"}),
    ("github__search_code", "在 GitHub 上搜索代码片段", {"query": "搜索关键词"}),
    ("github__list_commits", "列出仓库的提交记录", {"owner": "仓库所有者", "repo": "仓库名", "sha": "分支或提交 SHA"}),
    ("github__fork_repository", "把仓库 fork 到自己的账号下", {"owner": "仓库所有者", "repo": "仓库名"}),
    # ---- Linear ----
    ("linear__create_issue", "在 Linear 中创建一个 issue", {"team_id": "团队 ID", "title": "标题", "description": "描述"}),
    ("linear__list_issues", "列出 Linear 的 issue", {"team_id": "团队 ID", "assignee": "负责人"}),
    ("linear__get_issue", "获取 Linear issue 详情", {"issue_id": "issue ID"}),
    ("linear__update_issue", "更新 Linear issue 的属性", {"issue_id": "issue ID", "state": "状态"}),
    ("linear__list_projects", "列出 Linear 项目", {"team_id": "团队 ID"}),
    ("linear__create_project", "创建一个 Linear 项目", {"name": "项目名", "team_id": "团队 ID"}),
    ("linear__list_teams", "列出当前账号加入的团队", {}),
    ("linear__add_comment", "给 Linear issue 添加评论", {"issue_id": "issue ID", "body": "评论内容"}),
    # ---- Jira ----
    ("jira__create_issue", "在 Jira 中创建工单", {"project_key": "项目键", "summary": "摘要", "issue_type": "工单类型"}),
    ("jira__get_issue", "获取 Jira 工单详情", {"issue_key": "工单键，如 PROJ-123"}),
    ("jira__update_issue", "更新 Jira 工单字段", {"issue_key": "工单键", "fields": "要更新的字段"}),
    ("jira__search_issues", "用 JQL 搜索 Jira 工单", {"jql": "JQL 查询语句"}),
    ("jira__transition_issue", "把 Jira 工单流转到下一个状态", {"issue_key": "工单键", "transition_id": "流转 ID"}),
    ("jira__list_transitions", "列出某个工单可用的流转", {"issue_key": "工单键"}),
    ("jira__add_comment", "给 Jira 工单添加评论", {"issue_key": "工单键", "body": "评论内容"}),
    ("jira__assign_issue", "把 Jira 工单指派给某人", {"issue_key": "工单键", "assignee": "被指派人"}),
    # ---- Slack ----
    ("slack__send_message", "向 Slack 频道发送消息", {"channel": "频道 ID", "text": "消息内容"}),
    ("slack__list_channels", "列出工作区的 Slack 频道", {}),
    ("slack__get_channel_history", "获取频道的历史消息", {"channel": "频道 ID", "limit": "条数"}),
    ("slack__create_channel", "创建一个 Slack 频道", {"name": "频道名"}),
    ("slack__invite_user", "邀请用户加入频道", {"channel": "频道 ID", "user": "用户 ID"}),
    ("slack__upload_file", "上传文件到 Slack 频道", {"channel": "频道 ID", "file": "文件路径"}),
    ("slack__add_reaction", "给消息添加表情回应", {"channel": "频道 ID", "timestamp": "消息时间戳", "emoji": "表情名"}),
    ("slack__search_messages", "搜索 Slack 消息", {"query": "搜索关键词"}),
    # ---- Notion ----
    ("notion__create_page", "在 Notion 中创建页面", {"parent_id": "父页面或数据库 ID", "title": "标题"}),
    ("notion__update_page", "更新 Notion 页面属性", {"page_id": "页面 ID", "properties": "属性"}),
    ("notion__get_page", "获取 Notion 页面内容", {"page_id": "页面 ID"}),
    ("notion__search", "搜索 Notion 中的页面和数据库", {"query": "搜索关键词"}),
    ("notion__list_databases", "列出可访问的 Notion 数据库", {}),
    ("notion__query_database", "按条件查询 Notion 数据库的记录", {"database_id": "数据库 ID", "filter": "过滤条件"}),
    ("notion__append_blocks", "向页面追加内容块", {"page_id": "页面 ID", "children": "内容块"}),
    # ---- Sentry ----
    ("sentry__list_issues", "列出 Sentry 的报错 issue", {"project_slug": "项目标识", "query": "过滤条件"}),
    ("sentry__get_issue", "获取 Sentry issue 详情", {"issue_id": "issue ID"}),
    ("sentry__resolve_issue", "把 Sentry issue 标记为已解决", {"issue_id": "issue ID"}),
    ("sentry__delete_issue", "删除 Sentry 的 issue", {"issue_id": "issue ID"}),
    ("sentry__list_projects", "列出 Sentry 项目", {}),
    ("sentry__get_event", "获取某个事件的堆栈详情", {"event_id": "事件 ID"}),
    # ---- Postgres ----
    ("postgres__query", "执行只读的 SQL 查询", {"sql": "SQL 语句"}),
    ("postgres__list_tables", "列出数据库中的所有表", {"schema": "模式名"}),
    ("postgres__describe_table", "查看表结构（列定义与索引）", {"table": "表名"}),
    ("postgres__insert_row", "向表中插入一行数据", {"table": "表名", "values": "列与值"}),
    ("postgres__delete_row", "删除表中的行", {"table": "表名", "where": "条件"}),
]


# (查询, 可接受的答案集合, 备注)
QUERIES: list[tuple[str, set[str], str]] = [
    ("创建一个 issue", {"github__create_issue", "linear__create_issue", "jira__create_issue"}, "三个平台都算对"),
    ("在 GitHub 上开一个工单", {"github__create_issue"}, "平台已限定"),
    ("读一下仓库里 README 的内容", {"github__get_file_contents"}, "文件读取"),
    ("提交一个 pull request", {"github__create_pull_request"}, "中文说 PR"),
    ("把这个 PR 合并掉", {"github__merge_pull_request"}, "合并语义"),
    ("帮我搜一下代码里哪里用了这个函数", {"github__search_code"}, "代码搜索"),
    ("列出最近的 commit 记录", {"github__list_commits"}, "提交历史"),
    ("把仓库 fork 一份到我的账号", {"github__fork_repository"}, "fork"),
    ("给 Jira 工单加个评论", {"jira__add_comment"}, "同 server 内有 add_comment 的还有 linear，但平台限定了"),
    ("把工单流转到下一个状态", {"jira__transition_issue"}, "工作流流转"),
    ("用 JQL 查一下这周的工单", {"jira__search_issues"}, "JQL"),
    ("在 Slack 频道里发条消息", {"slack__send_message"}, "发消息"),
    ("看看这个频道最近聊了什么", {"slack__get_channel_history"}, "频道历史"),
    ("邀请同事加入频道", {"slack__invite_user"}, "邀请"),
    ("在 Notion 里建一个页面", {"notion__create_page"}, "建页面"),
    ("按条件查询 Notion 数据库", {"notion__query_database"}, "数据库查询"),
    ("看 Sentry 最近有哪些报错", {"sentry__list_issues"}, "报错列表"),
    ("把这个报错标记为已解决", {"sentry__resolve_issue"}, "状态变更"),
    ("跑一条 SQL 看看数据", {"postgres__query"}, "SQL"),
    ("数据库里有哪些表", {"postgres__list_tables"}, "表清单"),
]


def build_tools() -> list[Any]:
    """把 `CATALOG` 变成 LangChain 工具对象（与 `build_docs` 的输入形状一致）。"""
    from langchain_core.tools import StructuredTool

    tools: list[Any] = []
    for name, desc, params in CATALOG:
        def _f(**_kwargs):
            return "ok"

        tools.append(
            StructuredTool.from_function(
                func=_f,
                name=name,
                description=desc,
                args_schema={
                    "type": "object",
                    "properties": {
                        k: {"type": "string", "description": v} for k, v in params.items()
                    },
                },
            )
        )
    return tools
