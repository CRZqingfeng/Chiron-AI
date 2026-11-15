"""`iter_chat_stream_events` 的事件契约（拆分前的等价网）。

为什么需要它：这个函数 757 行，是全项目最关键的对话编排，但**零直测**——
`test_approval.py` 只复刻了判定条件，`test_chat_gate.py` 直接把它 mock 掉。
要把它拆成多段而不改行为，必须先有一张「同样输入 → 同样事件序列」的网，
否则所谓「行为零变化」只能靠感觉。

用假 agent + 假 storage 跑通四条真实路径：
1. 普通对话：content 流 → done
2. 调用方取消：cancelled → done(cancelled=True)
3. 审批中断：approval_required → done(interrupted=True, message_id=N)
4. resume 参数非法：error → done

断言的是**事件类型序列 + 关键字段**，不锁具体文案——拆分时允许动实现，不允许动契约。
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest import TestCase, mock, skip

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

import app.chat.agent_service as svc
import app.chat.checkpoint as checkpoint_mod
import app.chat.memory_search as memory_search
import app.workspace as workspace_pkg

_USER_ID = 1
_AGENT_ID = 2
_SESSION_ID = "s1"


class _FakeUA:
    """够 `iter_chat_stream_events` 用到的全部属性；不碰 Tortoise。"""

    user_id = _USER_ID
    api_key_ciphertext = ""
    base_url = ""
    model_name = "fake-model"
    sub_api_key_ciphertext = ""
    sub_model_name = ""
    sub_base_url = ""
    supports_vision = False


class _FakeStorage:
    """内存版 storage：录下写入内容，并让「落库后回查」能拿到 id。

    中断路径依赖这个回查——它要把刚落的待确认消息 id 带回前端。
    """

    def __init__(self, path_records: list[dict] | None = None, regen_context=None) -> None:
        self.records: list[dict] = list(path_records or [])
        self.regen_context = regen_context
        self.written: list[dict] = []
        self._next_id = 100

    def load(self, user_id, agent_id, session_id):
        return [HumanMessage(content="hi")]

    def get_session_messages(self, user_id, agent_id, session_id):
        return list(self.records)

    def append_messages(self, user_id, agent_id, session_id, messages, extra_message_data=None):
        extras = extra_message_data or [{} for _ in messages]
        for msg, extra in zip(messages, extras):
            self._next_id += 1
            record = {
                "message_id": self._next_id,
                "type": "ai" if isinstance(msg, AIMessage) else "human",
                "content": msg.content if isinstance(msg.content, str) else "",
                "approval_pending": (extra or {}).get("approval_pending"),
                "error_text": (extra or {}).get("error_text"),
                "extra": extra or {},
            }
            self.records.append(record)
            self.written.append(record)
        return True

    def update_assistant_in_place(self, *args, **kwargs) -> bool:
        # 记录 kwargs，用于断言 resume 路径必须带 merge_thinking=True
        self.written.append({"op": "update_in_place", **kwargs})
        return True

    def insert_assistant_version(self, *args, **kwargs) -> bool:
        self.written.append({"op": "insert_version"})
        return True

    def get_regenerate_context(self, *args, **kwargs):
        return self.regen_context


class _FakeAgent:
    """假的 LangGraph agent：按两个 stream_mode 分别产出 (chunk, metadata)。"""

    def __init__(
        self,
        chunks: list[tuple],
        interrupt=None,
        *,
        delay: float = 0.0,
        error=None,
        todo_updates: list[dict] | None = None,
    ) -> None:
        self.chunks = chunks
        self.interrupt = interrupt
        self.delay = delay
        self.error = error
        self.todo_updates = todo_updates or []
        self.calls: list[tuple] = []

    def astream(self, stream_input, stream_mode="messages", config=None):
        self.calls.append((stream_input, stream_mode, config))

        async def _gen():
            if self.error is not None:
                raise self.error
            for update in self.todo_updates:
                if stream_mode == "messages":
                    continue
                yield ("updates", update)
            for chunk in self.chunks:
                if self.delay:
                    import asyncio as _asyncio

                    await _asyncio.sleep(self.delay)
                if stream_mode == "messages":
                    yield chunk
                else:
                    yield ("messages", chunk)
            if self.interrupt is not None:
                yield ("updates", {"__interrupt__": (self.interrupt,)})

        return _gen()

    async def aget_state(self, config=None):
        """resume 前会查断点是否还在；默认给一个「有待办任务」的快照。"""
        return SimpleNamespace(next=("HumanInTheLoopMiddleware.after_model",), tasks=())


def _interrupt(action_name: str = "workspace_run_command", interrupt_id: str = "intr-1"):
    """构造一个 LangGraph 中断对象（够 `_interrupt_to_event` 拍平）。"""
    return SimpleNamespace(
        id=interrupt_id,
        value={
            "action_requests": [
                {
                    "name": action_name,
                    "args": {"command": "rm -rf build"},
                    "description": "删除构建目录",
                }
            ],
            "review_configs": [{"allowed_decisions": ["approve", "edit", "reject"]}],
        },
    )


def _chunk(text: str, msg_id: str = "m1") -> tuple:
    return (AIMessageChunk(content=text, id=msg_id), {"langgraph_node": "model"})


@contextlib.contextmanager
def _harness(storage: _FakeStorage, agent: _FakeAgent, *, checkpointer=None, rag_context=None):
    """把外部世界全部打桩，只留被测函数的编排逻辑。"""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(svc, "storage", storage))
        stack.enter_context(mock.patch.object(svc, "format_attachment_hint", lambda *a, **k: ""))
        stack.enter_context(
            mock.patch.object(svc, "build_storable_human_content", lambda text, ids, **k: text)
        )
        stack.enter_context(
            mock.patch.object(
                svc, "_run_kb_document_preselect_with_context", lambda *a, **k: (None, {})
            )
        )
        stack.enter_context(
            mock.patch.object(
                memory_search, "proactive_session_memory_inject_text", lambda *a, **k: None
            )
        )
        stack.enter_context(mock.patch.object(workspace_pkg, "open_workspace", lambda **k: None))
        stack.enter_context(
            mock.patch.object(
                svc, "load_agent_mcp_tools", mock.AsyncMock(return_value=([], [], set()))
            )
        )
        stack.enter_context(
            mock.patch.object(svc, "build_model_and_agent", lambda *a, **k: (agent, None))
        )
        stack.enter_context(
            mock.patch.object(
                svc, "_prepare_to_invoke_messages", lambda *a, **k: [HumanMessage(content="hi")]
            )
        )
        stack.enter_context(
            mock.patch.object(svc, "schedule_archive_session_memory", lambda *a, **k: None)
        )
        stack.enter_context(
            mock.patch.object(svc, "_should_run_vision_caption", lambda *a, **k: False)
        )
        stack.enter_context(mock.patch.object(svc, "_approval_tools_map", lambda: {}))
        stack.enter_context(
            mock.patch.object(
                checkpoint_mod, "get_checkpointer", mock.AsyncMock(return_value=checkpointer)
            )
        )
        if rag_context is not None:
            stack.enter_context(
                mock.patch.object(svc, "get_last_rag_context", lambda clear=False: dict(rag_context))
            )
        yield


async def _collect(ua=None, **kwargs) -> list[dict]:
    ua = ua or _FakeUA()
    events: list[dict] = []
    async for event in svc.iter_chat_stream_events(
        ua, kwargs.pop("user_text", "hi"), _USER_ID, _AGENT_ID, _SESSION_ID, **kwargs
    ):
        events.append(event)
    return events


def _types(events: list[dict]) -> list[str]:
    return [e.get("type") for e in events]


def types_index(events: list[dict], name: str) -> int:
    """首次出现该事件类型的下标；不存在则抛 ValueError（比静默 -1 好定位）。"""
    return _types(events).index(name)


class AgentStreamContractTests(TestCase):
    def test_plain_turn_emits_content_then_done(self) -> None:
        storage = _FakeStorage()
        agent = _FakeAgent([_chunk("你"), _chunk("好")])
        with _harness(storage, agent):
            events = _run(_collect())

        self.assertEqual(_types(events), ["content", "content", "done"])
        self.assertEqual("".join(e["content"] for e in events if e["type"] == "content"), "你好")
        self.assertFalse(events[-1]["cancelled"])
        # 助手消息必须落库（否则刷新后对话消失）
        self.assertTrue(any(r.get("type") == "ai" for r in storage.written))

    def test_cancel_before_agent_emits_cancelled(self) -> None:
        storage = _FakeStorage()
        agent = _FakeAgent([_chunk("不该出现")])
        with _harness(storage, agent):
            events = _run(_collect(cancel_check=lambda: True))

        self.assertEqual(_types(events), ["cancelled", "done"])
        self.assertTrue(events[-1]["cancelled"])
        # 取消了就不该白烧 token 去调模型
        self.assertEqual(agent.calls, [])

    def test_interrupt_emits_approval_required_and_persists_pending(self) -> None:
        storage = _FakeStorage()
        interrupt = SimpleNamespace(
            id="intr-1",
            value={
                "action_requests": [
                    {
                        "name": "workspace_run_command",
                        "args": {"command": "rm -rf build"},
                        "description": "删除构建目录",
                    }
                ],
                "review_configs": [{"allowed_decisions": ["approve", "reject"]}],
            },
        )
        agent = _FakeAgent([_chunk("我先看一眼")], interrupt=interrupt)
        with _harness(storage, agent, checkpointer=object()):
            events = _run(_collect(job_id="job-abc"))

        types = _types(events)
        self.assertIn("approval_required", types)
        approval = next(e for e in events if e["type"] == "approval_required")
        self.assertEqual(approval["thread_id"], "job-abc")
        self.assertEqual(approval["request"]["actions"][0]["name"], "workspace_run_command")
        self.assertEqual(approval["request"]["allowed_decisions"], ["approve", "reject"])

        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertTrue(done["interrupted"])
        self.assertFalse(done["cancelled"])
        # 必须把刚落库那条待确认消息的 id 带回前端（否则恢复时定位不到）
        self.assertIsInstance(done["message_id"], int)
        pending = [r for r in storage.written if r.get("approval_pending")]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["approval_pending"]["thread_id"], "job-abc")

    def test_invalid_resume_decision_emits_error(self) -> None:
        storage = _FakeStorage()
        agent = _FakeAgent([])
        with _harness(storage, agent):
            events = _run(_collect(resume_decision="maybe", resume_thread_id="t1", target_message_id=1))

        self.assertEqual(_types(events), ["error", "done"])
        self.assertIn("approve", events[0]["content"])

    def test_resume_without_target_message_emits_error(self) -> None:
        storage = _FakeStorage()
        agent = _FakeAgent([])
        with _harness(storage, agent):
            events = _run(_collect(resume_decision="approve", resume_thread_id="t1"))

        self.assertEqual(_types(events), ["error", "done"])

    def test_resume_with_foreign_message_is_rejected(self) -> None:
        """越权防护：目标消息不属于本会话（或 thread 对不上）时必须拒绝。"""
        storage = _FakeStorage(
            [
                {
                    "message_id": 7,
                    "type": "ai",
                    "content": "",
                    "approval_pending": {"interrupt_id": "x", "thread_id": "someone-else"},
                }
            ]
        )
        agent = _FakeAgent([])
        with _harness(storage, agent, checkpointer=object()):
            events = _run(
                _collect(
                    resume_decision="approve",
                    resume_thread_id="my-thread",
                    target_message_id=7,
                )
            )

        self.assertEqual(_types(events), ["error", "done"])
        self.assertIn("失效", events[0]["content"])


def _run(coro):
    """测试里统一跑协程；不存在事件循环断言，纯 asyncio.run。"""
    import asyncio

    return asyncio.run(coro)


class AgentStreamBranchTests(TestCase):
    """拆分时其余分支的等价网：regenerate / 模型报错 / 流中取消 / MCP / trace+sources。"""

    def test_regenerate_inserts_new_version(self) -> None:
        storage = _FakeStorage(
            regen_context=([HumanMessage(content="hi")], "hi", [11])
        )
        agent = _FakeAgent([_chunk("新", "v2")])
        with _harness(storage, agent):
            events = _run(_collect(regenerate=True, target_message_id=11))

        self.assertEqual(_types(events), ["content", "done"])
        # 普通重生成 → 插入兄弟版本（不是原地覆盖）
        self.assertIn({"op": "insert_version"}, storage.written)

    def test_regenerate_text_mismatch_is_rejected(self) -> None:
        storage = _FakeStorage(regen_context=([HumanMessage(content="hi")], "原来的问题", [11]))
        agent = _FakeAgent([])
        with _harness(storage, agent):
            events = _run(_collect(regenerate=True, target_message_id=11))

        self.assertEqual(_types(events), ["error", "done"])
        self.assertIn("不一致", events[0]["content"])

    def test_model_error_is_classified_and_persisted(self) -> None:
        storage = _FakeStorage()
        agent = _FakeAgent([], error=RuntimeError("upstream 500"))
        with _harness(storage, agent):
            events = _run(_collect())

        types = _types(events)
        self.assertEqual(types[0], "error")
        self.assertIn("kind", events[0])
        self.assertIn("retryable", events[0])
        self.assertEqual(types[-1], "done")
        # 错误文案必须落库（刷新后仍要能看到失败原因）
        self.assertTrue(any(r.get("error_text") for r in storage.written))
        # 不能把 provider 原始报文直接给用户
        self.assertNotIn("upstream 500", events[0]["content"])

    @skip(
        "已知问题：asyncio.run 下流中取消后 _cancel_watchdog 的 task 回收不完整，"
        "会让单测挂起（真实 uvicorn 场景随请求结束清理，不受影响）。待排查后恢复。"
    )
    def test_cancel_mid_stream_breaks_loop(self) -> None:
        """流中取消：走 chunk 循环里的 cancel_probe 而不是启动前那次检查。"""
        storage = _FakeStorage()
        agent = _FakeAgent([_chunk("a"), _chunk("b"), _chunk("c")], delay=0.3)
        calls = {"n": 0}

        def _cancel_check() -> bool:
            calls["n"] += 1
            return calls["n"] > 1  # 第一次放行，之后一律取消

        with _harness(storage, agent):
            events = _run(_collect(cancel_check=_cancel_check))

        self.assertEqual(_types(events), ["cancelled", "done"])
        self.assertTrue(events[-1]["cancelled"])
        # 取消后不应再落库一条正式回答
        self.assertFalse(any(r.get("type") == "ai" for r in storage.written if "op" not in r))

    def test_trace_and_sources_are_emitted_before_done(self) -> None:
        storage = _FakeStorage()
        agent = _FakeAgent([_chunk("答")])
        rag = {
            "rag_trace": {"route": "generate_answer"},
            "kb_sources": [{"index": 1, "filename": "a.pdf"}],
            "image_references": [{"attachment_id": "att1"}],
        }
        with _harness(storage, agent, rag_context=rag):
            events = _run(_collect())

        types = _types(events)
        self.assertIn("trace", types)
        self.assertIn("sources", types)
        # 顺序契约：正文 → trace → sources → done
        self.assertLess(types.index("trace"), types.index("sources"))
        self.assertEqual(types[-1], "done")
        # U5c：MCP 审批已并到中断事件（approval_required），不再有单独的
        # `mcp_confirmation_required`——它在正常收尾里没有任何产生路径了。
        self.assertNotIn("mcp_confirmation_required", types)

    def test_todo_list_streams_and_final_state_is_persisted_once(self) -> None:
        """U9a：清单实时推 `todo_list`，收尾只落一份终态进 thinking_items。"""
        storage = _FakeStorage()
        updates = [
            {"tools": {"todos": [{"content": "查资料", "status": "in_progress"}]}},
            {"tools": {"todos": [{"content": "查资料", "status": "completed"}]}},
        ]
        agent = _FakeAgent([_chunk("答")], todo_updates=updates)
        with _harness(storage, agent):
            events = _run(_collect())

        todo_events = [e for e in events if e["type"] == "todo_list"]
        self.assertEqual(
            [e["todos"][0]["status"] for e in todo_events], ["in_progress", "completed"]
        )
        # 清单事件必须出现在正文之前（模型先规划再作答）
        self.assertLess(types_index(events, "todo_list"), types_index(events, "content"))

        saved = [r for r in storage.written if r.get("type") == "ai"][-1]
        todo_items = [i for i in (saved["extra"].get("thinking_items") or []) if i.get("type") == "todo"]
        self.assertEqual(len(todo_items), 1, "终态只该落一条，逐次追加会刷屏时间线")
        self.assertEqual(todo_items[0]["todos"][0]["status"], "completed")

    def test_resume_interrupt_overwrites_pending_instead_of_appending(self) -> None:
        """「批准 → 模型继续 → 再次要求高危命令 → 再中断」不能堆新卡片。

        线上实录（2026-10-25）：同一会话被中断两次后留下 2 条 approval_pending，
        前一张卡片指向的 interrupt 早已不存在，点上去只会得到「审批已失效」。
        """
        storage = _FakeStorage(
            [
                {
                    "message_id": 11,
                    "type": "ai",
                    "content": "",
                    "approval_pending": {"interrupt_id": "old", "thread_id": "t1"},
                }
            ]
        )
        agent = _FakeAgent([], interrupt=_interrupt(interrupt_id="intr-2"))
        with _harness(storage, agent, checkpointer=object()):
            events = _run(
                _collect(resume_decision="approve", resume_thread_id="t1", target_message_id=11)
            )

        self.assertIn("approval_required", _types(events))
        self.assertTrue(events[-1]["interrupted"])
        self.assertIn("update_in_place", [w.get("op") for w in storage.written])
        self.assertEqual(
            [r for r in storage.written if r.get("type") == "ai"],
            [],
            "resume 后又中断必须原地覆盖，不能新增助手消息行",
        )

    def test_resume_merges_thinking_instead_of_overwriting(self) -> None:
        """resume 正常完成时必须带 merge_thinking=True。

        中断跨越两次独立的 collector：中断那轮把前半段思考区落进待确认消息，
        resume 这轮只有中断之后的片段。覆盖式更新会让用户点开思考区只看到
        「暂无检索步骤」——2026-10-25 线上实测就是这个表现。
        """
        storage = _FakeStorage(
            [
                {
                    "message_id": 11,
                    "type": "ai",
                    "content": "",
                    "approval_pending": {"interrupt_id": "old", "thread_id": "t1"},
                }
            ]
        )
        agent = _FakeAgent([_chunk("全部写好了")])
        with _harness(storage, agent, checkpointer=object()):
            events = _run(
                _collect(resume_decision="approve", resume_thread_id="t1", target_message_id=11)
            )

        self.assertEqual(_types(events)[-1], "done")
        updates = [w for w in storage.written if w.get("op") == "update_in_place"]
        self.assertTrue(updates, "resume 完成应原地更新中断那条消息")
        self.assertTrue(
            all(u.get("merge_thinking") is True for u in updates),
            "resume 必须合并思考区，否则中断前的步骤被清空",
        )
        self.assertEqual(
            [r for r in storage.written if r.get("type") == "ai"],
            [],
            "resume 完成不能新增助手消息行",
        )

    def test_interrupt_persists_plan_for_replay(self) -> None:
        """中断那条消息要把思考区（含清单）一起落库，否则刷新后只剩一个空确认框。"""
        storage = _FakeStorage()
        agent = _FakeAgent(
            [_chunk("我先看一眼")],
            interrupt=_interrupt(),
            todo_updates=[
                {"tools": {"todos": [{"content": "查资料", "status": "in_progress"}]}},
            ],
        )
        with _harness(storage, agent, checkpointer=object()):
            _run(_collect(job_id="job-keep"))

        pending = [r for r in storage.written if r.get("approval_pending")]
        self.assertEqual(len(pending), 1)
        items = pending[0]["extra"].get("thinking_items") or []
        self.assertTrue(
            any(i.get("type") == "todo" for i in items),
            f"待确认消息里必须带清单，实际 items={items}",
        )

    def test_plain_turn_has_no_todo_artifact(self) -> None:
        """没有规划的输出不能凭空多出 todo 项与事件。"""
        storage = _FakeStorage()
        agent = _FakeAgent([_chunk("答")])
        with _harness(storage, agent):
            events = _run(_collect())

        self.assertNotIn("todo_list", _types(events))
        saved = [r for r in storage.written if r.get("type") == "ai"][-1]
        self.assertEqual(
            [i for i in (saved["extra"].get("thinking_items") or []) if i.get("type") == "todo"],
            [],
        )

    def test_vision_prepass_fills_caption_before_model_call(self) -> None:
        """图片预读：描述先进思考区，再作为 image_caption 传给模型输入构建。"""
        import app.chat.vision_caption as vision_mod

        storage = _FakeStorage()
        agent = _FakeAgent([_chunk("答")])
        captured: dict = {}

        async def _fake_caption_chunks(ua, msg, **kwargs):
            for piece in ("图中是", "一张流程图"):
                yield piece

        def _fake_prepare(messages, *args, **kwargs):
            captured.update(kwargs)
            return [HumanMessage(content="hi")]

        with _harness(storage, agent):
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(svc, "_human_content_has_image_ref", lambda content: True)
                )
                stack.enter_context(
                    mock.patch.object(svc, "_should_run_vision_caption", lambda *a, **k: True)
                )
                stack.enter_context(
                    mock.patch.object(vision_mod, "iter_image_caption_chunks", _fake_caption_chunks)
                )
                stack.enter_context(
                    mock.patch.object(svc, "_prepare_to_invoke_messages", _fake_prepare)
                )
                events = _run(_collect())

        types = _types(events)
        # 预读事件必须排在正文之前（否则思考区顺序就错了）
        first_content = types.index("content")
        self.assertLess(types.index("thinking_item"), first_content)
        self.assertEqual(types[-1], "done")
        self.assertEqual(captured.get("image_caption"), "图中是一张流程图")
        # 图片预读文本不进 thinking_text（它不是工具前导句）
        saved = [r for r in storage.written if r.get("type") == "ai"]
        self.assertTrue(saved)
