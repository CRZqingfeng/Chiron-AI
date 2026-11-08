"""测试类命令的会话级授权（U10 P1-E）。

**要解决的问题**：`pytest -x -q` 每跑一次都要点一次确认，用户会被烦到把审批整个关掉——
那才是真正的安全损失（能力全景 §9 记的失效模式）。

**授权语义**：批准过一次，**本会话内**同类命令不再问。
不是「一律放行」：换会话、换用户、换智能体都要重新确认一次。

**为什么不能并进只读白名单**：pytest 会加载 `conftest.py`，而那是模型可写的文件。
免审批的 pytest + 可写的 conftest，等于给了一条绕过 `python script.py` 审批的通道。
所以它必须有「用户批准过」这个前置。

**存储**：Redis，key 带会话作用域，24 小时过期。Redis 不可用时一律返回「未授权」，
方向是保守的（多问一次，而不是静默放行）。
"""

from __future__ import annotations

from app.core.cache import cache
from app.log import logger

# 授权有效期。会话本身可能活很久，但「允许跑测试」这件事不该永久有效——
# 用户换了会话、换了任务，重新确认一次是合理的
GRANT_TTL_SECONDS = 24 * 60 * 60

_KEY_PREFIX = "chat_test_cmd_grant"


def _key(scope: str | None) -> str:
    return f"{_KEY_PREFIX}:{scope or ''}"


def grant_test_commands(scope: str | None) -> bool:
    """记录「本会话允许跑测试类命令」。作用域为空时不做任何事。"""
    if not scope:
        return False
    try:
        return bool(cache.set_json(_key(scope), {"granted": True}, ttl=GRANT_TTL_SECONDS))
    except Exception as e:  # noqa: BLE001 - 授权写失败只影响体验，不影响安全
        logger.warning("测试命令授权写入失败（下次仍需确认）")
        logger.debug("授权写异常详情：{}", e)
        return False


def test_commands_granted(scope: str | None) -> bool:
    """本会话是否已授权跑测试。拿不到作用域或 Redis 不可用都返回 False。"""
    if not scope:
        return False
    try:
        data = cache.get_json(_key(scope))
    except Exception as e:  # noqa: BLE001
        logger.debug("测试命令授权读取失败（按未授权处理）：{}", e)
        return False
    return bool(isinstance(data, dict) and data.get("granted"))


def revoke_test_commands(scope: str | None) -> None:
    """撤销授权（会话删除或用户主动收回时用）。"""
    if not scope:
        return
    try:
        cache.delete(_key(scope))
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "GRANT_TTL_SECONDS",
    "grant_test_commands",
    "revoke_test_commands",
    "test_commands_granted",
]
