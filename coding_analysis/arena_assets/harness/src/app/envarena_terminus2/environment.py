import logging
import os
import shutil
from pathlib import Path

import anyio
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


class LocalEnvironment(BaseEnvironment):
    def __init__(self, session_id: str, trial_root: Path = Path("/logs")) -> None:
        self.environment_dir = Path("/")
        self.environment_name = "envarena-task-container"
        self.session_id = session_id
        self.context_id = None
        self.trial_paths = TrialPaths(trial_root)
        self.task_env_config = EnvironmentConfig(workdir=os.getcwd())
        self.extra_docker_compose_paths = []
        self.default_user = None
        self.logger = logging.getLogger("EnvArenaTerminus2")

    @staticmethod
    def type() -> str:
        return "envarena-local"

    def _validate_definition(self) -> None:
        return None

    async def start(self, force_build: bool) -> None:
        return None

    async def stop(self, delete: bool) -> None:
        return None

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        _copy_file(Path(source_path), Path(target_path))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        _copy_file(Path(source_path), Path(target_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        process_env = dict(os.environ)
        process_env.update(env or {})
        try:
            with anyio.fail_after(timeout_sec):
                result = await anyio.run_process(
                    ["/bin/bash", "-c", command],
                    cwd=cwd,
                    env=process_env,
                    check=False,
                )
        except TimeoutError:
            return ExecResult(
                stderr=f"command timed out after {timeout_sec}s", return_code=124
            )
        return ExecResult(
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            return_code=result.returncode,
        )


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
