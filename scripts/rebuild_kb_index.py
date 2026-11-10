"""知识库索引重建脚本（Milvus collection 改名后使用）。

从 MinIO 读取原始文件字节，逐个重跑入库流水线，写入新的 collection。
不改动文档元数据本身；`run_ingest_pipeline_sync` 内部走「先处理后替换」语义。

用法：
    venv/Scripts/python.exe .tmp/rebuild_kb.py --limit 1     # 先试一个最小的
    venv/Scripts/python.exe .tmp/rebuild_kb.py               # 全量
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import object_storage as obs  # noqa: E402
from app.core.db import SessionLocal  # noqa: E402
from app.kb.db_models import KbDocument  # noqa: E402
from app.kb.kb_service import agent_kb_key_prefix, run_ingest_pipeline_sync  # noqa: E402


def parse_scope(scope: str) -> tuple[int, int]:
    m = re.match(r"u(\d+)_a(\d+)", scope or "")
    if not m:
        raise ValueError(f"无法解析 kb_scope: {scope}")
    return int(m.group(1)), int(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（按 chunk 数升序）")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        rows = [
            (d.kb_scope, d.display_filename, d.stored_filename, d.chunk_count or 0)
            for d in db.query(KbDocument).all()
        ]
    finally:
        db.close()

    rows.sort(key=lambda r: r[3])
    if args.limit:
        rows = rows[: args.limit]

    print(f"待重建 {len(rows)} 个文档（按 chunk 数升序）\n")
    ok = fail = 0

    for scope, name, stored, chunks in rows:
        try:
            uid, aid = parse_scope(scope)
            key = obs.join_key(agent_kb_key_prefix(uid, aid), stored)
            content = obs.read_bytes(key)
            print(f"▶ {scope} / {name}（原文 {len(content)} 字节，原 {chunks} chunks）")

            state = {"stage": ""}

            def cb(stage, done, total):
                if stage != state["stage"]:
                    state["stage"] = stage
                    print(f"    {stage} ...")

            r = run_ingest_pipeline_sync(
                kb_scope=scope,
                user_id=uid,
                agent_id=aid,
                display_filename=name,
                content=content,
                progress_cb=cb,
            )
            flag = "未变跳过" if r.get("unchanged") else "已重建"
            print(f"✅ {name} [{flag}] -> {r.get('chunk_count')} chunks / {r.get('parent_chunks')} parents\n")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"❌ {scope}/{name}: {type(e).__name__}: {e}\n")
            traceback.print_exc()
            fail += 1

    print(f"===== 完成：成功 {ok}，失败 {fail} =====")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
