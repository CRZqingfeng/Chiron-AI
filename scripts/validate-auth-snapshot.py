#!/usr/bin/env python
"""鉴权快照缓存（Redis 白名单）验收脚本。

用法：
    venv/Scripts/python.exe scripts/validate-auth-snapshot.py

设计前提：**不触碰你正在运行的 :9999 实例**。本脚本另起两个临时实例：
    :10099  正常（读 .env 的 Redis）
    :10098  故意指向一个没有监听的 Redis 端口，用于验证降级
结束时自动回收两个进程，并删除脚本自建的测试用户。

覆盖范围：
   A 接线完整性（静态）：配置项、白名单字段、三处必须失效的调用点
   B 端到端：
     B2  首次请求后快照落盘
     B3  连续请求 TTL 单调递减 —— 证明全程命中、从未回源查库
     B3b 快照毒化 —— 证明判定确实采信快照而非偷偷查库
     B4  改密 → 旧 token 立即 401 且快照键被删
     B5  新密码登录可用
     B6  管理员禁用 → 立即 401
     B7  管理员重新启用 → 新登录立即恢复 200（原方案会漏的正确性 bug）
     B8  管理员删除 → 立即 401
     B9  Redis 不可用 → 同一 token 仍 200（fail-closed 降级），坏 token 仍 401
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx
import redis

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

# 直接以脚本方式运行时，sys.path[0] 是 scripts/，需把项目根加进来才能 import app
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_PORT = 10099
DEGRADED_PORT = 10098
DEAD_REDIS_URL = "redis://127.0.0.1:6399/0"  # 未监听端口，用于降级验证

ADMIN_USER = "admin"
ADMIN_PASS = "Chiron2026ai"
TEST_PASS = "ChironSnap2026init"
TEST_PASS2 = "ChironSnap2026changed"
HDR = "token"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    ok = bool(ok)
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   -- {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{title}")


def has_live_call(segment: str, name: str) -> bool:
    """段内是否存在「非注释行」上的 name( 调用。

    变异测试发现：只做 `name in segment` 会被注释掉的那一行骗过
    （`# invalidate_user(...)` 里的子串照样命中），故必须排除注释行。
    """
    return live_line_index(segment, f"{name}(") != -1


def live_line_index(segment: str, needle: str) -> int:
    """首个「非注释行」中包含 needle 的行号（用于断言调用先后顺序），无则 -1。

    用「包含」而非「行首前缀」匹配：调用常带前缀，如 `await user_controller.bump_auth_epoch(user)`。
    """
    for i, line in enumerate(segment.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and needle in stripped:
            return i
    return -1


def env_value(key: str, default: str = "") -> str:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or default
    return os.environ.get(key, default)


# --------------------------------------------------------------------------- #
# Part A 静态接线
# --------------------------------------------------------------------------- #
def part_a() -> None:
    section("Part A 接线完整性（静态）")
    import app  # noqa: F401  确保能正常导入（含新增模块）

    from app.settings import settings
    from app.utils.auth_cache import SNAPSHOT_FIELDS, SNAPSHOT_VERSION, _key, _ttl

    check("config 暴露 AUTH_SNAPSHOT_TTL_SECONDS", hasattr(settings, "AUTH_SNAPSHOT_TTL_SECONDS"),
          f"= {getattr(settings, 'AUTH_SNAPSHOT_TTL_SECONDS', None)}s")
    check("快照字段白名单不含 password", "password" not in SNAPSHOT_FIELDS, f"{SNAPSHOT_FIELDS}")
    check("快照字段白名单不含关系字段 roles", "roles" not in SNAPSHOT_FIELDS)
    check("快照 key 形如 auth_snap:{id}", _key(7) == "auth_snap:7", _key(7))
    check("快照结构带版本号", SNAPSHOT_VERSION >= 1, f"v={SNAPSHOT_VERSION}")
    check("TTL 取自配置", _ttl() == settings.AUTH_SNAPSHOT_TTL_SECONDS, f"{_ttl()}s")

    dep = (ROOT / "app/core/dependency.py").read_text(encoding="utf-8")
    check("dependency.is_authed 已接入快照读取", "get_snapshot(" in dep and "user_from_snapshot(" in dep)
    check("dependency.is_authed 未命中会回写快照", "cache_user(" in dep)
    check("dependency.is_authed 不再无条件查库",
          "user_from_snapshot(get_snapshot(user_id))" in dep)

    ctl = (ROOT / "app/controllers/user.py").read_text(encoding="utf-8")
    bump = ctl.split("async def bump_auth_epoch", 1)[-1]
    check("bump_auth_epoch 会失效快照", has_live_call(bump, "invalidate_user"))
    check("bump_auth_epoch 先失效快照再吊销 refresh",
          live_line_index(bump, "invalidate_user(") != -1
          and live_line_index(bump, "revoke_user_refresh_tokens(") != -1
          and live_line_index(bump, "invalidate_user(")
          < live_line_index(bump, "revoke_user_refresh_tokens("))

    usr = (ROOT / "app/api/v1/users/users.py").read_text(encoding="utf-8")
    upd = usr.split("async def update_user", 1)[-1].split("@router", 1)[0]
    dele = usr.split("async def delete_user", 1)[-1]
    check("users.update_user 无条件失效快照", has_live_call(upd, "invalidate_user"))
    check("users.update_user 的失效在 bump 判断之前",
          live_line_index(upd, "invalidate_user(") != -1
          and live_line_index(upd, "bump_auth_epoch") != -1
          and live_line_index(upd, "invalidate_user(")
          < live_line_index(upd, "bump_auth_epoch"))
    check("users.delete_user 会失效快照", has_live_call(dele, "invalidate_user"))

    base = (ROOT / "app/api/v1/base/base.py").read_text(encoding="utf-8")
    avatar = base.split("async def upload_avatar", 1)[-1].split("@router", 1)[0]
    check("base.upload_avatar 写头像后失效快照", has_live_call(avatar, "invalidate_user"))


# --------------------------------------------------------------------------- #
# 进程与 HTTP 辅助
# --------------------------------------------------------------------------- #
def spawn(port: int, extra_env: dict | None = None) -> tuple[subprocess.Popen, object]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)
    code = f"import uvicorn; uvicorn.run('app:app', host='127.0.0.1', port={port}, log_config=None)"
    log = open(Path(tempfile.gettempdir()) / f"chiron-validate-{port}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(PY), "-c", code], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT
    )
    return proc, log


def wait_health(port: int, timeout: int = 240) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/api/v1/base/health", timeout=3.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def make_client(port: int) -> httpx.Client:
    return httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)


def login(client: httpx.Client, username: str, password: str) -> str | None:
    r = client.post("/api/v1/base/access_token", json={"username": username, "password": password})
    if r.status_code != 200:
        print(f"      登录失败 {username}: HTTP {r.status_code} {r.text[:200]}")
        return None
    return r.json().get("data", {}).get("access_token")


def probe(client: httpx.Client, token: str) -> httpx.Response:
    return client.get("/api/v1/base/userinfo", headers={HDR: token})


# --------------------------------------------------------------------------- #
# Part B 端到端
# --------------------------------------------------------------------------- #
def part_b(main_port: int) -> str | None:
    prefix = env_value("REDIS_KEY_PREFIX", "chiron_ai")
    rds = redis.Redis.from_url(env_value("REDIS_URL", "redis://127.0.0.1:6379/0"), decode_responses=True)
    ttl_expected = int(env_value("AUTH_SNAPSHOT_TTL_SECONDS", "0") or 0) or 300

    admin = make_client(main_port)
    user = make_client(main_port)

    tag = uuid.uuid4().hex[:8]
    email = f"snap-{tag}@example.com"
    uname = f"snap{tag}"
    admin_token: str | None = None
    test_uid: int | None = None

    try:
        # ---- 准备：管理员登录 + 建测试用户 ----
        section("Part B 准备")
        admin_token = login(admin, ADMIN_USER, ADMIN_PASS)
        if not check("管理员登录", admin_token is not None):
            return None

        r = admin.post(
            "/api/v1/user/create",
            headers={HDR: admin_token},
            json={"email": email, "username": uname, "password": TEST_PASS,
                  "is_active": True, "role_ids": [], "dept_id": 0},
        )
        created = r.status_code == 200 and (r.json().get("data") or {}).get("id")
        if not check("创建测试用户", bool(created), f"HTTP {r.status_code} {r.text[:160]}"):
            return admin_token
        test_uid = int(r.json()["data"]["id"])
        snap_key = f"{prefix}:auth_snap:{test_uid}"
        rds.delete(snap_key)

        token = login(user, email, TEST_PASS)
        if not check("测试用户登录", token is not None):
            return admin_token

        # ---- B1~B3 命中路径 ----
        section("Part B 命中路径（不查库）")
        check("首请求前快照不存在", rds.exists(snap_key) == 0)

        rp = probe(user, token)
        check("B1 首请求放行 200", rp.status_code == 200, f"HTTP {rp.status_code}")

        ttl1 = rds.ttl(snap_key)
        check("B2 未命中后快照已回写", ttl1 > 0, f"TTL={ttl1}s")

        time.sleep(3)
        codes = [probe(user, token).status_code for _ in range(3)]
        ttl2 = rds.ttl(snap_key)
        check("B3 连续 3 次请求全部 200", codes == [200, 200, 200], f"{codes}")
        check(
            "B3 TTL 单调递减（未发生 miss，即未回源查库）",
            ttl2 > 0 and ttl2 <= ttl1 - 2,
            f"{ttl1}s -> {ttl2}s；一旦 miss 会重置为 {ttl_expected}s。"
            f"cache_user() 只在 miss 分支被调用，故 TTL 不减即证明 User.filter() 未执行",
        )

        # ---- B3b 快照毒化：证明判定确实采信快照 ----
        section("Part B 快照权威性（毒化验证）")
        rds.setex(
            snap_key, 60,
            json.dumps({"v": 1, "id": test_uid, "username": uname, "email": email,
                        "is_active": True, "is_superuser": False,
                        "token_version": 999, "dept_id": 0}, ensure_ascii=False),
        )
        rp = probe(user, token)
        check("B3b 毒化快照（token_version=999）导致 401",
              rp.status_code == 401
              and ("过期" in rp.text),
              f"HTTP {rp.status_code} {rp.text[:120]}；若此时真去查了库，DB 里 tv=0 与 token 相符，会误判 200")

        rds.delete(snap_key)
        rp = probe(user, token)
        check("B3b 删键后回源 DB 立即恢复 200", rp.status_code == 200, f"HTTP {rp.status_code}")

        # ---- B4/B5 改密 ----
        section("Part B 改密即时吊销")
        r = user.post("/api/v1/base/update_password", headers={HDR: token},
                      json={"old_password": TEST_PASS, "new_password": TEST_PASS2})
        check("改密接口返回 200", r.status_code == 200, f"{r.text[:160]}")
        check("B4 改密后快照键被删除", rds.exists(snap_key) == 0, snap_key)

        rp = probe(user, token)
        check("B4 改密后旧 token 立即 401", rp.status_code == 401,
              f"HTTP {rp.status_code} {rp.text[:120]}")

        token2 = login(user, email, TEST_PASS2)
        if check("B5 新密码可登录", token2 is not None):
            check("B5 新 token 放行 200", probe(user, token2).status_code == 200)
        check("B5 旧密码已不可登录", login(user, email, TEST_PASS) is None)

        # ---- B6/B7 禁用 / 重新启用 ----
        section("Part B 禁用与重新启用")
        body = {"id": test_uid, "email": email, "username": uname,
                "is_active": False, "role_ids": [], "dept_id": 0}
        r = admin.post("/api/v1/user/update", headers={HDR: admin_token}, json=body)
        check("管理员禁用用户", r.status_code == 200, f"{r.text[:160]}")
        check("B6 禁用后快照键被删除", rds.exists(snap_key) == 0)
        if token2:
            rp = probe(user, token2)
            check("B6 禁用后 token 立即 401", rp.status_code == 401,
                  f"HTTP {rp.status_code} {rp.text[:120]}")
        raw = rds.get(snap_key) or ""
        check("B6 缓存的禁用快照 is_active=false（回源时写入）",
              '"is_active": false' in raw, raw[:140])

        body["is_active"] = True
        r = admin.post("/api/v1/user/update", headers={HDR: admin_token}, json=body)
        check("管理员重新启用", r.status_code == 200, f"{r.text[:160]}")
        check("B7 重新启用后快照键被删除（本次修复点）", rds.exists(snap_key) == 0)

        token3 = login(user, email, TEST_PASS2)
        if check("B7 重新启用后可登录", token3 is not None):
            rp = probe(user, token3)
            check("B7 重新启用后立即恢复 200",
                  rp.status_code == 200,
                  f"HTTP {rp.status_code} {rp.text[:120]}"
                  f"　← 若无条件失效缺失，此处会命中残留的 is_active=false 快照而误判 401")

        # ---- B8 删除 ----
        section("Part B 删除用户")
        r = admin.delete(f"/api/v1/user/delete?user_id={test_uid}", headers={HDR: admin_token})
        check("管理员删除用户", r.status_code == 200, f"{r.text[:160]}")
        check("B8 删除后快照键不存在", rds.exists(snap_key) == 0)
        if token3:
            rp = probe(user, token3)
            check("B8 删除后 token 立即 401", rp.status_code == 401,
                  f"HTTP {rp.status_code} {rp.text[:120]}")
        test_uid = None

    finally:
        if test_uid is not None and admin_token:
            try:
                admin.delete(f"/api/v1/user/delete?user_id={test_uid}",
                             headers={HDR: admin_token})
                print(f"      (已清理遗留测试用户 id={test_uid})")
            except Exception:
                pass
        admin.close()
        user.close()

    return admin_token


def part_b9(main_token: str | None, degraded_port: int) -> None:
    section("Part B Redis 不可用降级")
    if not main_token:
        check("B9 跳过：无可用 token", False)
        return
    good = make_client(API_PORT)
    bad = httpx.Client(base_url=f"http://127.0.0.1:{degraded_port}", timeout=30.0)
    try:
        check("正常实例健康检查可达", good.get("/api/v1/base/health").status_code == 200)
        check("降级实例健康检查可达", bad.get("/api/v1/base/health").status_code == 200)

        r = bad.get("/api/v1/base/userinfo", headers={HDR: main_token})
        check("B9 Redis 不可用时合法 token 仍 200（自动回源查库）",
              r.status_code == 200, f"HTTP {r.status_code} {r.text[:120]}")
        r = bad.get("/api/v1/base/userinfo", headers={HDR: "not-a-jwt"})
        check("B9 Redis 不可用时坏 token 仍 401（未退化成全放行）",
              r.status_code == 401, f"HTTP {r.status_code}")
    finally:
        good.close()
        bad.close()


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 78)
    print("Chiron-AI 鉴权快照缓存验收")
    print(f"项目根: {ROOT}")
    print("=" * 78)

    part_a()

    procs: list[tuple[subprocess.Popen, object]] = []
    try:
        section(f"Part B 启动测试实例 :{API_PORT}（不影响你正在运行的 :9999）")
        p, log = spawn(API_PORT)
        procs.append((p, log))
        if not wait_health(API_PORT):
            print(f"      启动失败，日志见 {log.name}")
            print(Path(log.name).read_text(encoding="utf-8", errors="replace")[-3000:])
            check("测试实例启动", False)
            return 1
        check("测试实例启动并就绪", True)

        admin_token = part_b(API_PORT)

        section(f"Part B 启动降级实例 :{DEGRADED_PORT}（REDIS_URL -> 死端口）")
        p2, log2 = spawn(DEGRADED_PORT, {"REDIS_URL": DEAD_REDIS_URL, "AUTH_RATE_LIMIT_ENABLED": "false"})
        procs.append((p2, log2))
        if not wait_health(DEGRADED_PORT, timeout=240):
            print(f"      启动失败，日志见 {log2.name}")
            print(Path(log2.name).read_text(encoding="utf-8", errors="replace")[-3000:])
            check("降级实例启动", False)
        else:
            check("降级实例启动并就绪", True)
            part_b9(admin_token, DEGRADED_PORT)
    finally:
        for p, log in procs:
            try:
                p.terminate()
                p.wait(timeout=20)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            try:
                log.close()
            except Exception:
                pass

    print("\n" + "=" * 78)
    print(f"结果：PASS {len(PASSED)} / FAIL {len(FAILED)}")
    if FAILED:
        print("失败项：")
        for n in FAILED:
            print(f"  - {n}")
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
