"""把 admin 账号密码重置为 .env 里的 INITIAL_ADMIN_PASSWORD。

背景：`INITIAL_ADMIN_PASSWORD` 只在**首次建库**时生效；项目更名后该值变了，
但数据库里仍是旧密码的 argon2 哈希，于是用新密码登录会失败。
跑一次本脚本即可把库里的哈希同步成 .env 当前值。
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg  # noqa: E402

from app.settings.config import _tortoise_pg_url, settings  # noqa: E402
from app.utils.password import get_password_hash, validate_password_strength  # noqa: E402


async def main() -> int:
    pwd = (settings.INITIAL_ADMIN_PASSWORD or "").strip()
    user = (settings.INITIAL_ADMIN_USERNAME or "admin").strip()
    if not pwd:
        print("❌ .env 未配置 INITIAL_ADMIN_PASSWORD")
        return 1

    validate_password_strength(pwd)
    hashed = get_password_hash(pwd)

    conn = await asyncpg.connect(_tortoise_pg_url(settings.chat_database_url))
    try:
        result = await conn.execute(
            'UPDATE "user" SET password = $1 WHERE username = $2', hashed, user
        )
        print(f"UPDATE 结果: {result}（目标用户 {user}）")
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
