"""端到端对话测试：登录 -> 发起对话 Job -> 轮询直到完成 -> 打印回复。

用于改名后的回归验证，确认 API Key 解密、LLM 调用、工作区等链路都正常。
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx

# 允许从任意 cwd 执行：把项目根加入 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.settings.config import settings  # noqa: E402

BASE = f"http://127.0.0.1:{settings.UVICORN_PORT}/api/v1"
STATUS_DONE = {"succeeded", "success", "completed", "done", "finished", "failed", "error", "cancelled", "canceled"}


def main() -> int:
    c = httpx.Client(timeout=30)

    # ① 登录（账号密码取自 .env，避免硬编码）
    user = (settings.INITIAL_ADMIN_USERNAME or "admin").strip()
    pwd = (settings.INITIAL_ADMIN_PASSWORD or "").strip()
    if not pwd:
        print("❌ .env 未配置 INITIAL_ADMIN_PASSWORD")
        return 1
    r = c.post(f"{BASE}/base/access_token", json={"username": user, "password": pwd})
    if r.status_code != 200:
        print(f"❌ 登录失败: {r.status_code} {r.text[:200]}")
        return 1
    token = r.json()["data"]["access_token"]
    h = {"token": token}
    print(f"① 登录成功（{user}）")

    # ② 取智能体列表
    r = c.get(f"{BASE}/user-agent/list", headers=h)
    agents = r.json().get("data") or []
    if not agents:
        print("❌ 没有可用智能体")
        return 1
    print(f"② 智能体: {[(a['id'], a['name']) for a in agents]}")
    aid = agents[0]["id"]

    # ③ 发起对话
    msg = "你好，用一句话介绍你自己"
    r = c.post(f"{BASE}/user-agent/chat/jobs", headers=h, json={"agent_id": aid, "message": msg})
    print(f"③ 创建对话 job: HTTP {r.status_code} {r.text[:200]}")
    if r.status_code != 200:
        return 1
    data = r.json().get("data") or {}
    jid = data.get("job_id") or data.get("id")
    if not jid:
        print(f"❌ 响应里没有 job_id: {data}")
        return 1
    print(f"   job_id = {jid}")

    # ④ 轮询状态
    last = None
    for i in range(75):
        time.sleep(2)
        sr = c.get(f"{BASE}/user-agent/chat/jobs/{jid}", headers=h)
        d = sr.json().get("data") or {}
        status = str(d.get("status") or d.get("state") or "").lower()
        if status != last:
            print(f"   [{i * 2:3d}s] status={status}")
            last = status
        if status in STATUS_DONE:
            print("\n=== 最终响应 ===")
            print(json.dumps(d, ensure_ascii=False)[:1500])
            if status in {"failed", "error"}:
                return 1
            # 取回复文本做一次存在性判断
            text = json.dumps(d, ensure_ascii=False)
            return 0 if len(text) > 200 else 1
    print("⏰ 超时：150 秒内未完成")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
