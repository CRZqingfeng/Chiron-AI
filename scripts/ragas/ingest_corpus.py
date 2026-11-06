# -*- coding: utf-8 -*-
"""把一组本地文档灌进指定智能体的知识库（走项目自己的入库流水线）。

为什么直接调 run_ingest_pipeline_sync 而不走 HTTP 上传：
上传接口要登录态与 multipart，脚本里模拟成本高且容易和前端行为不一致；
这个函数就是 upload 接口最终调用的同一个东西（kb_job 里的入口），口径一致。

用法：
  ./.venv/Scripts/python.exe scripts/ragas/ingest_corpus.py \
      --dir=.tmp/ragas/corpus2 --agent-id=3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="待入库文件所在目录")
    ap.add_argument("--agent-id", type=int, required=True)
    ap.add_argument("--user-id", type=int, default=1)
    a = ap.parse_args()

    from app.kb.kb_scope import kb_scope_for
    from app.kb.kb_service import allowed_upload_extension, run_ingest_pipeline_sync
    from app.kb.milvus_client import MilvusManager

    MilvusManager().init_collection()
    scope = kb_scope_for(a.user_id, a.agent_id)
    files = sorted(p for p in Path(a.dir).iterdir() if p.is_file())
    if not files:
        raise SystemExit("目录为空")

    print(f"kb_scope={scope}  待入库 {len(files)} 份", flush=True)
    for f in files:
        if not allowed_upload_extension(f.name):
            print(f"  跳过（不支持的扩展名）: {f.name}")
            continue
        try:
            meta = run_ingest_pipeline_sync(
                kb_scope=scope,
                user_id=a.user_id,
                agent_id=a.agent_id,
                display_filename=f.name,
                content=f.read_bytes(),
            )
        except Exception as e:
            print(f"  ✗ {f.name}: {type(e).__name__}: {e}", flush=True)
            continue
        chunks = meta.get("chunk_count") if isinstance(meta, dict) else None
        flag = "（内容未变，跳过重建）" if isinstance(meta, dict) and meta.get("unchanged") else ""
        print(f"  ✓ {f.name}: chunks={chunks} {flag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
