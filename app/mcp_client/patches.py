"""MCP 域自己的 schema / 数据补丁。

为什么单独成文件：这条补丁要读 `app.mcp_client.presets` 里的只读服务 url 列表，
而 `app/core/schema_patches.py` 是「通用列变更」的登记处，不该知道 MCP 的存在。
core 提供了 `register_patches()`，这里在 import 时把自己的补丁登记进去。

导入时机：由组合根 `app/init_app.py` 在调用 `apply_schema_patches` 之前 import，
否则补丁不会被执行（会静默跳过——这也是把它放在组合根而不是 core 里的原因）。
"""

from __future__ import annotations

from app.core.schema_patches import SchemaPatch, register_patches


def _preset_confirm_policy_statements() -> list[str]:
    """已知只读 MCP 预置的存量 auto 配置收紧为 never（避免 Context7 等查询服务被误拦）。"""
    from app.mcp_client.presets import MCP_SERVER_PRESETS

    readonly_urls = [p["url"] for p in MCP_SERVER_PRESETS if p.get("confirm_policy") == "never"]
    if not readonly_urls:
        return []
    quoted = ", ".join("'" + u.replace("'", "''") + "'" for u in readonly_urls)
    return [
        "UPDATE user_agent_mcp_server SET confirm_policy = 'never'"
        f" WHERE confirm_policy = 'auto' AND url IN ({quoted})"
    ]


register_patches(
    SchemaPatch(
        name="100_mcp_preset_confirm_policy",
        description="只读 MCP 预置的存量 auto 配置收紧为 never（数据修正）",
        statements_fn=_preset_confirm_policy_statements,
    )
)
