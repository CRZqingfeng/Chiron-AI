"""Docker 实现：命令在一次性容器里跑，文件通过宿主挂载目录共享。

**关键简化**：因为工作区目录以 bind mount 挂进了容器，宿主与容器看到的是同一份
文件系统，所以 `read_text` / `write_text` / `list_dir` 直接复用 `LocalWorkspace`
的宿主实现——语义完全一致，而且比 `docker cp` / `docker exec cat` 快一个数量级
（每次 exec 都要起一个进程）。

只有 `execute` 必须进容器，那是隔离的意义所在。

容器策略（每轮一个）：
- `--rm`：正常退出即删，异常退出也不残留
- `--name chiron-ws-{job_id}`：便于 `cleanup` 与残留清扫
- `--memory/--cpus/--pids-limit`：单容器跑飞不能拖垮宿主
- `--network none`（默认）：当前只有文件类工具，不需要网

已知限制：`docker exec` 被宿主侧超时 kill 后，容器内可能有孤儿进程存活。
因此命令本身再包一层 `timeout`，双保险；容器最终随 cleanup 销毁，不会长期泄漏。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

from app.workspace.base import (
    CommandResult,
    WorkspaceError,
    WorkspaceUnavailable,
    truncate_text,
)
from app.workspace.local import LocalWorkspace, _decode

CONTAINER_PREFIX = "chiron-ws-"
CONTAINER_WORKDIR = "/workspace"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")
_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


def _sanitize_id(raw: str) -> str:
    """容器名只允许字母数字与 `_-`，其余替换为 `-`。

    job_id 理论上都是 uuid hex，但这里是"用户可控输入进 shell 参数"的边界，
    不能依赖上游一定干净。
    """
    cleaned = _SAFE_ID.sub("-", str(raw or ""))[:48]
    return cleaned or "anon"


def docker_available() -> bool:
    """探测 Docker 服务是否可用（CLI 在但 daemon 没起来也算不可用）。"""
    if not shutil.which("docker"):
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def reap_stale_containers(prefix: str = CONTAINER_PREFIX) -> int:
    """清理前缀匹配的残留容器，返回清理数量。

    进程被强杀时 `--rm` 来不及执行，容器会一直挂着占用内存与磁盘。
    在创建新工作区前顺手扫一遍，成本是一次 `docker ps`。

    ⚠️ 这个函数会**连带删掉正在运行的**同前缀容器，所以只能在工作区创建之前调用。
    `prefix` 参数是为测试留的（传一个不存在的唯一前缀即可验证无副作用）。
    """
    if not docker_available():
        return 0
    try:
        listed = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name={prefix}"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    ids = [x for x in listed.stdout.decode("utf-8", "replace").split() if x]
    if not ids:
        return 0
    try:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    return len(ids)


class DockerWorkspace(LocalWorkspace):
    """在一次性容器中执行命令；文件读写复用宿主的挂载目录实现。"""

    def __init__(
        self,
        root: Path | str,
        *,
        job_id: str,
        image: str = "python:3.11-slim",
        mem_limit: str = "512m",
        cpu_limit: float = 1.0,
        pids_limit: int = 128,
        network: bool = False,
        exec_timeout: float = 30.0,
        output_limit: int = 32_000,
        read_limit: int = 200_000,
        name_prefix: str = CONTAINER_PREFIX,
    ) -> None:
        super().__init__(
            root,
            exec_timeout=exec_timeout,
            output_limit=output_limit,
            read_limit=read_limit,
        )
        self.container = f"{name_prefix}{_sanitize_id(job_id)}"
        self._image = image
        self._mem_limit = mem_limit
        self._cpu_limit = float(cpu_limit)
        self._pids_limit = int(pids_limit)
        self._network = bool(network)
        self._started = False
        self._start_container()

    # --- 容器生命周期 -----------------------------------------------------

    def _start_container(self) -> None:
        if not docker_available():
            raise WorkspaceUnavailable("Docker 不可用")
        args = [
            "docker", "run", "-d", "--rm",
            "--name", self.container,
            # 资源上限：一个 fork bomb 或死循环不能拖垮宿主
            "--memory", self._mem_limit,
            "--cpus", str(self._cpu_limit),
            "--pids-limit", str(self._pids_limit),
            # 挂载：宿主与容器共享同一目录，文件方法因此可以直接走宿主
            "-v", f"{self.root}:{CONTAINER_WORKDIR}",
            "-w", CONTAINER_WORKDIR,
        ]
        # 默认断网：当前只有文件类工具，联网留给 U5b 的 run_command 按需开启
        args += ["--network", "bridge" if self._network else "none"]
        args += [self._image, "sleep", "infinity"]

        try:
            proc = subprocess.run(args, capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise WorkspaceUnavailable(f"启动容器失败：{e}") from e
        if proc.returncode != 0:
            err = _decode(proc.stderr).strip()[:500]
            raise WorkspaceUnavailable(f"启动容器失败：{err}")

        # 容器刚起来时 exec 偶发失败，探一次再放行，避免首个工具调用就报错
        if not self._wait_ready(timeout=10.0):
            self.cleanup()
            raise WorkspaceUnavailable("容器启动后未就绪")
        self._started = True

    def _wait_ready(self, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                probe = subprocess.run(
                    ["docker", "exec", self.container, "true"],
                    capture_output=True,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                probe = None
            if probe is not None and probe.returncode == 0:
                return True
            time.sleep(0.2)
        return False

    # --- 命令执行 ---------------------------------------------------------

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        # cwd 仍然先在宿主侧校验，保证容器内的路径也不会越界
        work = self._resolve(cwd) if cwd else self.root
        if not work.exists() or not work.is_dir():
            raise WorkspaceError(f"工作目录不存在：{cwd}")
        rel = work.relative_to(self.root).as_posix()
        container_cwd = CONTAINER_WORKDIR if rel == "." else f"{CONTAINER_WORKDIR}/{rel}"
        limit = float(timeout if timeout is not None else self._exec_timeout)

        # 容器内再包一层 timeout：宿主侧 kill 只杀 docker exec 进程，
        # 容器里的实际命令会变成孤儿继续跑；内层 timeout 才能杀掉它。
        args = [
            "docker", "exec",
            "-w", container_cwd,
            self.container,
            "timeout", f"{max(1, int(limit))}", "sh", "-c", command,
        ]
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                timeout=limit + 5,  # 宿主侧多给 5s，让容器内 timeout 先动手
            )
        except subprocess.TimeoutExpired as e:
            out, t_out = truncate_text(_decode(e.stdout or b""), self._output_limit)
            err, t_err = truncate_text(_decode(e.stderr or b""), self._output_limit)
            return CommandResult(
                stdout=out,
                stderr=(err + f"\n命令执行超时（{limit:.0f}s），已终止").strip(),
                exit_code=-1,
                duration_ms=int((time.perf_counter() - started) * 1000),
                truncated=t_out or t_err,
            )
        except OSError as e:
            raise WorkspaceError(f"容器执行失败：{e}") from e

        out, t_out = truncate_text(_decode(proc.stdout), self._output_limit)
        err, t_err = truncate_text(_decode(proc.stderr), self._output_limit)
        # coreutils `timeout` 超时的固定退出码是 124。这时容器内命令已被杀掉，
        # 但 stderr 是空的——若不补一句说明，模型只会看到一个孤零零的 124，
        # 无法判断「是命令本身失败还是跑太久被掐断」。
        if proc.returncode == 124:
            err = (err + f"\n命令执行超时（{limit:.0f}s），已终止").strip()
            return CommandResult(
                stdout=out,
                stderr=err,
                exit_code=-1,
                duration_ms=int((time.perf_counter() - started) * 1000),
                truncated=t_out or t_err,
            )
        return CommandResult(
            stdout=out,
            stderr=err,
            exit_code=int(proc.returncode),
            duration_ms=int((time.perf_counter() - started) * 1000),
            truncated=t_out or t_err,
        )

    # --- 清理 -------------------------------------------------------------

    def cleanup(self) -> None:
        """强制删除容器；挂载目录保留（跨轮持久）。幂等。"""
        self._started = False
        try:
            subprocess.run(
                ["docker", "rm", "-f", self.container],
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            # 清理失败不该影响调用方（generator 的 finally 里执行）
            pass


__all__ = ["DockerWorkspace", "docker_available", "reap_stale_containers", "CONTAINER_PREFIX"]
