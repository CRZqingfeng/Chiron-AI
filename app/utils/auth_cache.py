"""登录态快照缓存：用 Redis 白名单快照替代「每请求查 User 表」。

改这里的代码之前，请先读完这三条约定：

1. **缓存的是「用户快照」，不是「已吊销的 token」。**
   快照 TTL 到期只会导致未命中 → 回源查库（fail-closed）；
   反过来若缓存黑名单（存"已失效"），键过期会让已吊销的 token 复活（fail-open），
   那种设计必须保证 TTL ≥ 凭证寿命才安全。两种方向的失效语义完全相反。

2. **权威数据源始终是 DB，本模块只是加速层。**
   Redis 任何异常都退化成「未命中」，调用方无感，不影响正确性。
   因此本模块所有函数都不抛异常。

3. **任何写 user 行之后必须调用 `invalidate_user()`。**
   漏调不会立刻出错，只会让快照陈旧到 TTL 到期 —— 那种 bug 极难排查。
   当前必须在三处调用：`UserController.bump_auth_epoch`（改密/禁用/超管变更）、
   `users.update_user`（含"重新启用"，该分支不进 bump）、`users.delete_user`。

快照只含白名单标量字段、绝不含 password；关系字段（roles）不缓存，
因此改角色权限立即生效，无需失效本缓存。
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from app.chat.cache import cache
from app.settings import settings

# 快照结构版本：SNAPSHOT_FIELDS 变更时必须 +1，否则旧键缺字段会被读出脏数据
SNAPSHOT_VERSION = 1

# 参与鉴权判定或可能被 current_user 读取的标量字段；password / roles 永不入缓存
SNAPSHOT_FIELDS = (
    "id",
    "username",
    "email",
    "is_active",
    "is_superuser",
    "token_version",
    "dept_id",
)

_KEY = "auth_snap"


def _key(user_id: int) -> str:
    return f"{_KEY}:{int(user_id)}"


def _ttl() -> int:
    return max(1, int(getattr(settings, "AUTH_SNAPSHOT_TTL_SECONDS", 300) or 300))


def snapshot_of(user: Any) -> dict:
    """把 User 实例压成可 JSON 序列化的白名单字段快照。"""
    snap: dict = {f: getattr(user, f, None) for f in SNAPSHOT_FIELDS}
    snap["v"] = SNAPSHOT_VERSION
    return snap


def cache_user(user: Any) -> bool:
    """回写快照。失败只记日志（下次请求退回查库），不影响当前请求。"""
    try:
        return cache.set_json(_key(int(user.id)), snapshot_of(user), _ttl())
    except Exception as e:
        logger.warning("auth 快照写入失败: {}", e)
        return False


def get_snapshot(user_id: int) -> Optional[dict]:
    """读快照；未命中或 Redis 不可用均返回 None。"""
    return cache.get_json(_key(user_id))


def invalidate_user(user_id: int) -> None:
    """失效快照。写 user 行之后必须调用；下一次请求会回源查库并重建快照。"""
    cache.delete(_key(user_id))


def user_from_snapshot(snap: Any):
    """由快照在内存中重建 User；快照损坏或版本不符返回 None，调用方回源查库。"""
    if not isinstance(snap, dict) or int(snap.get("v") or 0) != SNAPSHOT_VERSION:
        return None
    from app.models import User

    user = User(**{f: snap.get(f) for f in SNAPSHOT_FIELDS})
    # 必须置 _saved_in_db=True：Tortoise 对未落库实例会拒绝执行关系查询
    # （tortoise/fields/relational.py: if not self.instance._saved_in_db: raise OperationalError），
    # 而 PermissionControl.has_permission 需要 `await current_user.roles`。
    # 该行确实存在于 DB 且主键齐全，置 True 是向 ORM 陈述事实，不是绕过校验。
    user._saved_in_db = True
    return user
