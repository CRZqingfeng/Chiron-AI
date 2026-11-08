"""沙箱镜像与命令白名单的一致性校验（U10 P1-B）。

**这份测试守的是一条承诺**：`command_policy._SAFE_BINARIES` 里的每个命令都免确认执行，
那它们就必须真的存在于镜像里。否则用户看到的是「没弹确认却失败了」，
第一反应是审批功能坏了。

所以清单**从白名单动态取**，不手抄：白名单里新增一个命令，这条测试立刻变红，
逼着人同步镜像（`deploy/sandbox.Dockerfile` 的自检 RUN 也要一起改）。

镜像不存在时整体跳过——不能让「没构建镜像」把测试套件搞红。
"""

from __future__ import annotations

import subprocess
import unittest

from app.workspace.command_policy import _SAFE_BINARIES

IMAGE_TAG = "chiron-sandbox:py311"


def _docker_ready() -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", IMAGE_TAG],
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


@unittest.skipUnless(
    _docker_ready(),
    f"{IMAGE_TAG} 未构建或 Docker 不可用（先跑 scripts/build_sandbox_image.py）",
)
class SandboxImageTests(unittest.TestCase):
    def _run(self, script: str) -> subprocess.CompletedProcess[str]:
        # 与生产一致：--network none + --rm，不给它任何联网或留存的机会
        return subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--memory",
                "512m",
                IMAGE_TAG,
                "sh",
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_every_whitelisted_binary_exists(self):
        names = " ".join(sorted(_SAFE_BINARIES))
        r = self._run(f"for b in {names}; do command -v $b >/dev/null || echo MISSING:$b; done")
        self.assertEqual(r.returncode, 0, r.stderr)
        missing = [ln for ln in r.stdout.splitlines() if ln.startswith("MISSING:")]
        self.assertEqual(missing, [], f"白名单承诺了但镜像里没有：{missing}")

    def test_pytest_is_available_offline(self):
        """沙箱断网，测试依赖必须烧进镜像，否则「脚本级验证」只能手写 assert。"""
        r = self._run("python -m pytest --version")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("pytest", r.stdout.lower())

    def test_git_can_run_offline(self):
        """白名单放行 `git status/diff/log`，所以 git 必须真能跑（不需要仓库也能 --version）。"""
        r = self._run("git --version")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("git version", r.stdout)

    def test_writes_into_workspace(self):
        """镜像默认用户必须能往挂载目录写——写文件是 P0 的主线能力。"""
        r = self._run("echo hi > /tmp/probe.txt && cat /tmp/probe.txt")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "hi")


if __name__ == "__main__":
    unittest.main()
