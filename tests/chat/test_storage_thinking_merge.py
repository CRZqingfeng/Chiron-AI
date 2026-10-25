"""审批续跑时思考区的合并规则（`ChatStorage._merge_thinking`）。

为什么单独测：审批中断跨越**两次独立的 collector**——中断那轮把前半段思考区
落进待确认消息，resume 这轮 collector 只有中断之后的片段。D-042 的两个修复
（A 原地覆盖 + B 落思考区）配合起来会出现「续跑把已有思考区覆盖成空」，
表现为用户点开思考区只看到「暂无检索步骤」。2026-10-25 现场复现过。

这些用例不需要数据库：`_merge_thinking` 只读 row 的属性 + 返回新 dict。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.chat.storage import ConversationStorage

merge = ConversationStorage._merge_thinking


def _row(thinking_items=None, thinking_text=None, rag_steps=None):
    return SimpleNamespace(
        thinking_items=thinking_items,
        thinking_text=thinking_text,
        rag_steps=rag_steps,
    )


def _fields(thinking_items=None, thinking_text=None, rag_steps=None):
    return {
        "thinking_items": thinking_items,
        "thinking_text": thinking_text,
        "rag_steps": rag_steps,
        "rag_trace": None,
        "error_text": None,
        "image_references": None,
        "sources": None,
        "approval_pending": None,
    }


class ThinkingMergeTests(unittest.TestCase):
    def test_new_only(self):
        """中断时没落思考区（旧代码产生的数据）：只用本轮内容。"""
        out = merge(_row(), _fields([{"type": "step", "label": "查库"}]))
        self.assertEqual(out["thinking_items"], [{"type": "step", "label": "查库"}])

    def test_old_kept_when_new_is_empty(self):
        """**核心用例**：续跑这轮除了正文没有别的步骤，已有思考区不能被清空。"""
        prev = [{"type": "step", "label": "检索"}, {"type": "text", "text": "我先查一下"}]
        out = merge(_row(thinking_items=prev), _fields(thinking_items=None))
        self.assertEqual(out["thinking_items"], prev)

    def test_concatenated_in_order(self):
        prev = [{"type": "step", "label": "检索"}]
        cur = [{"type": "step", "label": "写文件"}]
        out = merge(_row(thinking_items=prev), _fields(thinking_items=cur))
        self.assertEqual([x["label"] for x in out["thinking_items"]], ["检索", "写文件"])

    def test_both_empty_stays_empty(self):
        out = merge(_row(), _fields())
        self.assertIsNone(out["thinking_items"])

    def test_todo_keeps_only_last(self):
        """清单是整表替换语义：中断前一份 + 续跑后一份，回放只能看到最终那份。"""
        prev = [
            {"type": "step", "label": "列计划"},
            {"type": "todo", "todos": [{"content": "a", "status": "in_progress"}]},
        ]
        cur = [
            {"type": "step", "label": "写文件"},
            {"type": "todo", "todos": [{"content": "a", "status": "completed"}]},
        ]
        out = merge(_row(thinking_items=prev), _fields(thinking_items=cur))
        todos = [x for x in out["thinking_items"] if x["type"] == "todo"]
        self.assertEqual(len(todos), 1)
        self.assertEqual(todos[0]["todos"][0]["status"], "completed")
        # 位置在最后：中间的两条 step 都保留
        self.assertEqual(out["thinking_items"][-1]["type"], "todo")
        # 4 项拼起来、删掉 1 个旧 todo → 3 项；旧 todo 的位置在中间被移除
        self.assertEqual(len(out["thinking_items"]), 3)
        self.assertEqual(
            [x.get("label") for x in out["thinking_items"][:2]], ["列计划", "写文件"]
        )

    def test_thinking_text_concatenated(self):
        out = merge(_row(thinking_text="前半句"), _fields(thinking_text="后半句"))
        self.assertEqual(out["thinking_text"], "前半句后半句")

    def test_thinking_text_kept_when_new_empty(self):
        out = merge(_row(thinking_text="前半句"), _fields(thinking_text=None))
        self.assertEqual(out["thinking_text"], "前半句")

    def test_rag_steps_concatenated(self):
        prev = [{"icon": "🔍", "label": "检索"}]
        cur = [{"icon": "📄", "label": "读文件"}]
        out = merge(_row(rag_steps=prev), _fields(rag_steps=cur))
        self.assertEqual(len(out["rag_steps"]), 2)

    def test_other_fields_not_disturbed(self):
        """合并只动三个思考区字段，approval_pending 等必须原样透传（要清空它）。"""
        fields = _fields()
        fields["approval_pending"] = {"thread_id": "t1"}
        out = merge(_row(thinking_items=[{"type": "step", "label": "x"}]), fields)
        self.assertEqual(out["approval_pending"], {"thread_id": "t1"})
        self.assertIn("rag_trace", out)

    def test_malformed_items_ignored(self):
        """库里若有非 dict 的脏项（历史写入异常），不能让整段思考区炸掉。"""
        prev = [{"type": "step", "label": "ok"}, "garbage", None]
        out = merge(_row(thinking_items=prev), _fields(thinking_items=[{"type": "step"}]))
        self.assertEqual(len(out["thinking_items"]), 2)
        self.assertTrue(all(isinstance(x, dict) for x in out["thinking_items"]))


if __name__ == "__main__":
    unittest.main()
