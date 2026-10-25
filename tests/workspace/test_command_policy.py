"""只读命令白名单：判定表 + 与审批中间件的联动。

两个层次各有各的价值：

- `CommandPolicyTests` 断言「哪些算安全」—— 用例是给未来改规则的人看的合同。
- `SafeCommandAwareHITLTests` 断言「中间件真的按这个判定放行/拦截」—— 用真实
  `create_agent` + `InMemorySaver` 跑，不做 mock。策略对但没接上，等于没做。
"""

from __future__ import annotations

import unittest
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import PrivateAttr

from app.workspace.command_policy import is_safe_readonly_command


class CommandPolicyTests(unittest.TestCase):
    def _assert_safe(self, command: str) -> None:
        ok, why = is_safe_readonly_command(command)
        self.assertTrue(ok, f"应放行但被拦：{command!r} → {why}")

    def _assert_blocked(self, command: str) -> None:
        ok, why = is_safe_readonly_command(command)
        self.assertFalse(ok, f"应拦截但被放行：{command!r}")

    # ---- 放行：日常「看一眼」的只读命令 ----
    def test_common_readonly_commands_are_allowed(self):
        for cmd in (
            "ls",
            "ls -la",
            "dir",
            "pwd",
            "cat README.md",
            "head -n 20 app/main.py",
            "tail -100 log.txt",
            "wc -l app/main.py",
            'grep -rn "foo" app/',
            "grep -i foo x.txt",
            "grep -o foo x.txt",
            "git status",
            "git diff --stat",
            "git log --oneline -5",
            "git show HEAD",
            "git rev-parse HEAD",
            "sort data.txt",
            "sort -t, -k2 data.txt",
            "diff a.txt b.txt",
        ):
            with self.subTest(cmd=cmd):
                self._assert_safe(cmd)

    def test_git_version_is_allowed(self):
        self._assert_safe("git --version")

    # ---- 拦截：串命令 / 重定向 / 变量展开 / 通配符 ----
    def test_shell_metacharacters_are_blocked(self):
        for cmd in (
            "ls; rm -rf /",
            "ls | grep x",
            "echo hi && rm x",
            "cat a > b",
            "cat a >> b",
            "cat < b",
            "cat $(whoami)",
            "ls `id`",
            "ls\nrm x",
            "ls *.py",
            "ls ?.py",
            "cat ${HOME}/x",
        ):
            with self.subTest(cmd=cmd):
                self._assert_blocked(cmd)

    # ---- 拦截：二进制不在白名单 ----
    def test_dangerous_binaries_are_blocked(self):
        for cmd in (
            "rm -rf data",
            "mv a b",
            "cp a b",
            "touch x",
            "mkdir x",
            "chmod +x x",
            "bash -c id",
            "python -c print(1)",
            "python script.py",
            "curl http://x",
            "wget http://x",
            "sed -i s/x/y/ f",
            "awk {print}",
            "find . -delete",
            "find . -name x",
            "dd if=/dev/zero of=f",
            "tee out.txt",
        ):
            with self.subTest(cmd=cmd):
                self._assert_blocked(cmd)

    # ---- 拦截：危险参数 ----
    def test_write_flags_are_blocked(self):
        for cmd in (
            "git log --output=x.txt",
            "sort -o out.txt in.txt",
            "tree -o out.txt",
            "grep --in-place foo x",
        ):
            with self.subTest(cmd=cmd):
                self._assert_blocked(cmd)

    # ---- 拦截：想跑出工作区 ----
    def test_paths_escaping_workspace_are_blocked(self):
        for cmd in (
            "cat /etc/passwd",
            "cat ../../etc/passwd",
            "cat ~/.ssh/id_rsa",
            "cat C:/Windows/win.ini",
            "cat \\\\server\\share\\x",
        ):
            with self.subTest(cmd=cmd):
                self._assert_blocked(cmd)

    # ---- 拦截：状态变更类 git 子命令 ----
    def test_state_changing_git_subcommands_are_blocked(self):
        for cmd in (
            "git push",
            "git commit -m x",
            "git checkout main",
            "git reset --hard",
            "git clean -fd",
            "git branch -D main",
            "git remote add x y",
            "git config user.name x",
            "git stash",
            "git",
        ):
            with self.subTest(cmd=cmd):
                self._assert_blocked(cmd)

    def test_empty_and_oversized_are_blocked(self):
        self._assert_blocked("")
        self._assert_blocked("   ")
        self._assert_blocked("ls " + "a" * 400)


# ---------------------------------------------------------------------------
# 中间件联动：真实 create_agent + InMemorySaver
# ---------------------------------------------------------------------------


class _ScriptedChatModel(BaseChatModel):
    """按脚本依次返回 AIMessage。`bind_tools` 返回自身，否则 create_agent 会报错。"""

    script: List[AIMessage] = []
    _i: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = min(self._i, len(self.script) - 1)
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=self.script[idx])])


@tool
def workspace_run_command(command: str, cwd: Optional[str] = None, timeout: int = 30) -> str:
    """在工作区执行一条 shell 命令（测试替身，只回显参数）。"""
    return f"executed: {command}"


def _run_once(command: str) -> tuple[Any, list]:
    """跑一轮「模型要求执行命令」的完整流程，返回 (最终快照, 消息列表)。"""
    from langgraph.checkpoint.memory import InMemorySaver

    from langchain.agents import create_agent

    from app.chat.approval import SafeCommandAwareHITL

    model = _ScriptedChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "workspace_run_command", "args": {"command": command}, "id": "c1"}
                ],
            ),
            AIMessage(content="完成"),
        ]
    )
    agent = create_agent(
        model,
        tools=[workspace_run_command],
        middleware=[SafeCommandAwareHITL(interrupt_on={"workspace_run_command": True})],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": f"t-{abs(hash(command)) % 10**8}"}}
    out = agent.invoke({"messages": [HumanMessage("执行一下")]}, cfg)
    snap = agent.get_state(cfg)
    return snap, out["messages"]


class SafeCommandAwareHITLTests(unittest.TestCase):
    def test_safe_command_runs_without_interrupt(self):
        snap, messages = _run_once("ls -la")
        interrupts = [i for t in (snap.tasks or ()) for i in (getattr(t, "interrupts", ()) or ())]
        self.assertEqual(interrupts, [], "只读命令不该弹确认卡片")
        self.assertFalse(snap.next, "只读命令不该把图停在中途")
        tool_outputs = [str(getattr(m, "content", "")) for m in messages]
        self.assertTrue(
            any("executed: ls -la" in t for t in tool_outputs),
            f"工具应真的执行了，实际消息：{tool_outputs}",
        )

    def test_unsafe_command_still_interrupts(self):
        snap, messages = _run_once("rm -rf data")
        interrupts = [i for t in (snap.tasks or ()) for i in (getattr(t, "interrupts", ()) or ())]
        self.assertTrue(interrupts, "写/删类命令必须仍然中断等人确认")
        # 中断时工具尚未执行
        self.assertFalse(
            any("executed: rm -rf data" in str(getattr(m, "content", "")) for m in messages),
            "中断时不该已经执行命令",
        )

    def test_switch_off_falls_back_to_interrupt_everything(self):
        """关掉开关时，只读命令也要确认（与改造前行为一致）。"""
        from app.settings import settings

        from unittest import mock

        from app.chat import agent_service

        # 必须给一个能解出明文 key 的密文，否则 build_model_and_agent 在
        # 「未配置有效 API Key」处直接抛错，测不到装配
        from app.utils.api_key_crypto import encrypt_api_key

        ua = type(
            "UA",
            (),
            {
                "id": 1,
                "user_id": 1,
                "api_key_ciphertext": encrypt_api_key("sk-test"),
                "base_url": "https://api.openai.com/v1",
                "model_name": "gpt-4o-mini",
                "temperature": 0.3,
                "system_prompt": "你是助手",
                "enabled_mcp_servers": None,
                "knowledge_base_ids": None,
                "is_public": False,
                "description": "",
                "name": "t",
                "sub_api_key_ciphertext": None,
                "sub_base_url": None,
                "sub_model_name": None,
                "sub_temperature": None,
                "use_sub_model": False,
                "avatar": None,
            },
        )()
        with (
            mock.patch.object(settings, "CHAT_APPROVAL_SAFE_COMMANDS_ENABLED", False),
            mock.patch("langchain.agents.create_agent", return_value=mock.Mock()) as m,
        ):
            agent_service.build_model_and_agent(
                ua, 1, 1, "s1", interrupt_on={"workspace_run_command": True}, checkpointer=object()
            )
        names = [type(mw).__name__ for mw in (m.call_args.kwargs.get("middleware") or [])]
        self.assertIn("HumanInTheLoopMiddleware", names)
        self.assertNotIn("SafeCommandAwareHITL", names)


if __name__ == "__main__":
    unittest.main()
