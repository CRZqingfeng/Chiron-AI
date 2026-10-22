from typing import Optional

import jwt
from fastapi import Depends, Header, HTTPException, Request

from app.core.ctx import CTX_USER_ID
from app.models import Role, User
from app.settings import settings
from app.utils.auth_cache import cache_user, get_snapshot, user_from_snapshot


class AuthControl:
    @classmethod
    async def is_authed(cls, token: str = Header(..., description="token验证")) -> Optional["User"]:
        try:
            # 仅本地开发（DEBUG=true）允许免 JWT；生产环境勿开启 DEBUG
            if token == "dev":
                if not settings.DEBUG:
                    raise HTTPException(status_code=401, detail="无效的Token")
                user = await User.filter().first()
                if not user:
                    raise HTTPException(status_code=401, detail="无效的Token")
                if not user.is_active:
                    raise HTTPException(status_code=401, detail="用户已被禁用")
                CTX_USER_ID.set(int(user.id))
                return user
            decode_data = jwt.decode(token, settings.SECRET_KEY, algorithms=settings.JWT_ALGORITHM)
            user_id = int(decode_data.get("user_id") or 0)
            tv = int(decode_data.get("token_version", 0) or 0)

            # 快照命中 → 零 DB 查询；未命中（含 Redis 不可用）回源查库后回写。
            # is_authed 的返回契约是 User 实体（13 处 DependAuth 消费，PermissionControl
            # 还要 await user.roles），所以这里必须拿到 User 对象而非只比对版本号。
            user = user_from_snapshot(get_snapshot(user_id)) if user_id else None
            if user is None:
                user = await User.filter(id=user_id).first()
                if not user:
                    raise HTTPException(status_code=401, detail="无效的Token")
                cache_user(user)

            # 判定顺序与加缓存前保持一致：先 is_active 再 token_version
            if not user.is_active:
                raise HTTPException(status_code=401, detail="用户已被禁用")
            if tv != int(getattr(user, "token_version", 0) or 0):
                raise HTTPException(status_code=401, detail="登录已过期")
            CTX_USER_ID.set(int(user.id))
            return user
        except HTTPException:
            raise
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="登录已过期")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="无效的Token")
        except Exception:
            raise HTTPException(status_code=401, detail="无效的Token")


class PermissionControl:
    @classmethod
    async def has_permission(cls, request: Request, current_user: User = Depends(AuthControl.is_authed)) -> None:
        if current_user.is_superuser:
            return
        method = request.method
        path = request.url.path
        roles: list[Role] = await current_user.roles
        if not roles:
            raise HTTPException(status_code=403, detail="The user is not bound to a role")
        apis = [await role.apis for role in roles]
        permission_apis = list(set((api.method, api.path) for api in sum(apis, [])))
        if (method, path) not in permission_apis:
            raise HTTPException(status_code=403, detail=f"Permission denied method:{method} path:{path}")


DependAuth = Depends(AuthControl.is_authed)
DependPermission = Depends(PermissionControl.has_permission)
