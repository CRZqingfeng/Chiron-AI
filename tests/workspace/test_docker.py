"""DockerWorkspace 集成测：容器起停、挂载双向可见、资源限制、断网、超时。

Docker 不可用时整体 skip（CI 与本机都可能没装）。镜像缺失同样 skip，
避免测试因为要拉 130MB 镜像而卡住。
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

from app.workspace.docker import (
    CONTAINER_PREFIX,
    DockerWorkspace,
    docker_available,
    reap_stale_containers,
)

IMAGE = "python:3.11-slim"


def _image_ready() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "image", "inspect", IMAGE],
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(docker_available(), "Docker 不可用")
@unittest.skipUnless(_image_ready(), f"镜像 {IMAGE} 未就绪")
class DockerWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.job_id = "t" + uuid.uuid4().hex[:12]
        self.ws = DockerWorkspace(self.root, job_id=self.job_id)

    def tearDown(self) -> None:
        self.ws.cleanup()
        self._tmp.cleanup()

    def _container_running(self) -> bool:
        proc = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name=^{self.ws.container}$"],
            capture_output=True,
            timeout=15,
        )
        return bool(proc.stdout.strip())

    # --- 生命周期 ---------------------------------------------------------

    def test_container_runs_then_removed_on_cleanup(self):
        self.assertTrue(self._container_running(), "创建后容器应在运行")
        self.assertIn(CONTAINER_PREFIX, self.ws.container)
        self.ws.cleanup()
        self.assertFalse(self._container_running(), "cleanup 后容器应被删除")

    def test_cleanup_is_idempotent(self):
        self.ws.cleanup()
        self.ws.cleanup()  # 第二次不应抛异常

    # --- 宿主与容器共享目录 -----------------------------------------------

    def test_host_write_visible_in_container(self):
        self.ws.write_text("from_host.txt", "宿主写入")
        r = self.ws.execute("cat from_host.txt")
        self.assertEqual(r["exit_code"], 0)
        self.assertIn("宿主写入", r["stdout"])

    def test_container_write_visible_on_host(self):
        r = self.ws.execute("echo from_container > out.txt")
        self.assertEqual(r["exit_code"], 0)
        self.assertTrue((self.root / "out.txt").exists())
        self.assertIn("from_container", (self.root / "out.txt").read_text())

    def test_execute_cwd_subdir(self):
        self.ws.write_text("sub/a.txt", "x")
        r = self.ws.execute("pwd", cwd="sub")
        self.assertEqual(r["exit_code"], 0)
        self.assertIn("/workspace/sub", r["stdout"])

    # --- 隔离与限制 -------------------------------------------------------

    def test_memory_limit_applied(self):
        r = self.ws.execute("cat /sys/fs/cgroup/memory.max")
        if r["exit_code"] != 0:
            r = self.ws.execute("cat /sys/fs/cgroup/memory/memory.limit_in_bytes")
        self.assertEqual(r["exit_code"], 0, r["stderr"])
        value = int(r["stdout"].strip())
        self.assertEqual(value, 512 * 1024 * 1024, f"实际限制 {value}")

    def test_network_disabled(self):
        """`--network none`：DNS 应该失败，命令非 0 退出。"""
        r = self.ws.execute("python -c \"import socket;socket.gethostbyname('example.com')\"", timeout=20)
        self.assertNotEqual(r["exit_code"], 0)

    def test_timeout_kills_long_command(self):
        r = self.ws.execute("sleep 30", timeout=1)
        self.assertEqual(r["exit_code"], -1)
        self.assertIn("超时", r["stderr"])
        self.assertLess(r["duration_ms"], 15_000, "超时后应尽快返回，而不是等满 30s")

    def test_nonzero_exit_is_result_not_exception(self):
        r = self.ws.execute("exit 7")
        self.assertEqual(r["exit_code"], 7)

    def test_output_truncated(self):
        r = self.ws.execute("seq 1 20000", timeout=20)
        self.assertTrue(r["truncated"])
        self.assertIn("输出已截断", r["stdout"])


@unittest.skipUnless(docker_available(), "Docker 不可用")
class DockerProbeTests(unittest.TestCase):
    def test_docker_available_true_when_daemon_up(self):
        self.assertTrue(docker_available())

    def test_reap_returns_int(self):
        """用不存在的唯一前缀调用，验证返回类型且无副作用。

        不能直接按默认前缀调——那会把其他用例正在跑的容器一起删掉。
        """
        self.assertEqual(reap_stale_containers(prefix=f"chiron-ws-none-{uuid.uuid4().hex}"), 0)


if __name__ == "__main__":
    unittest.main()
