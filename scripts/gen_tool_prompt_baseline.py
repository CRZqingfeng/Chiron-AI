"""U6 改造前基线：4 组配置的 system prompt / turn context / 工具名。

改造后用同一脚本重跑，逐字节 diff —— 提示词是模型行为的直接输入，
改一个字都可能影响输出，所以这里必须比"看起来差不多"更严格。

用法：venv/Scripts/python.exe .u6_baseline.py > docs/dev/U6-baseline.md
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# 从 scripts/ 直接运行时要手动把项目根加入 sys.path，否则找不到 app 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.chat.agent_prompt import _compose_system_prompt, _format_turn_context_block
from app.utils.api_key_crypto import encrypt_api_key
from app.workspace.local import LocalWorkspace

# 配置矩阵：KB / web 两个开关的四种组合
CONFIGS = [
    ("A_kb_only", True, False),
    ("B_kb_web", True, True),
    ("C_web_only", False, True),
    ("D_neither", False, False),
]


def _ua():
    return SimpleNamespace(
        id=1,
        user_id=1,
        api_key_ciphertext=encrypt_api_key("sk-test"),
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        temperature=0.3,
        system_prompt="你是一个严谨的助手。",
        enabled_mcp_servers=None,
        knowledge_base_ids=None,
        is_public=False,
        description="",
        name="t",
        sub_api_key_ciphertext=None,
        sub_base_url=None,
        sub_model_name=None,
        sub_temperature=None,
        use_sub_model=False,
        avatar=None,
    )


def collect_tools(*, workspace: bool, interrupt: bool = False) -> list[str]:
    """拦截 create_agent 取实际注册的工具名。"""
    from app.chat import agent_service

    tmp = tempfile.TemporaryDirectory()
    kw = {}
    if workspace:
        kw["workspace"] = LocalWorkspace(Path(tmp.name))
    if interrupt:
        kw["checkpointer"] = mock.Mock()
        kw["interrupt_on"] = {"workspace_run_command": True}
    with mock.patch("langchain.agents.create_agent", return_value=mock.Mock()) as m:
        agent_service.build_model_and_agent(_ua(), 1, 1, "s1", **kw)
        names = sorted(t.name for t in (m.call_args.kwargs.get("tools") or []))
    tmp.cleanup()
    return names


def main() -> None:
    ua = _ua()
    out: list[str] = ["# U6 改造前基线（prompt 与工具名）", ""]

    out.append("## 1. system prompt（按配置）")
    for label, kb, web in CONFIGS:
        text = _compose_system_prompt(ua, use_knowledge_retrieval=kb, use_web_search=web)
        out.append(f"\n### {label}（kb={kb} web={web}）\n")
        out.append("```text")
        out.append(text)
        out.append("```")

    out.append("\n## 2. turn context block（按配置）")
    for label, kb, web in CONFIGS:
        text = _format_turn_context_block(
            use_knowledge_retrieval=kb,
            use_web_search=web,
            document_filter=None,
            session_attachment_hint="",
            memory_inject=None,
            mcp_approval_note=None,
            image_caption=None,
        )
        out.append(f"\n### {label}（kb={kb} web={web}）\n")
        out.append("```text")
        out.append(text)
        out.append("```")

    out.append("\n## 3. 工具名集合")
    variants = [
        ("默认（含 workspace）", dict(workspace=True)),
        ("默认（不含 workspace）", dict(workspace=False)),
        ("含审批（exec 注册）", dict(workspace=True, interrupt=True)),
    ]
    for label, kw in variants:
        names = collect_tools(**kw)
        out.append(f"\n### {label}\n")
        out.append(f"数量：{len(names)}")
        out.append("")
        out.append("```json")
        out.append(json.dumps(names, ensure_ascii=False, indent=2))
        out.append("```")

    print("\n".join(out))


if __name__ == "__main__":
    main()
