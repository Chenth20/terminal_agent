import json
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

from envarena_terminus2.trajectory import (
    IncompleteTrajectoryError,
    export_trajectory,
)

OUTER_DEADLINE_GUARD_SECONDS: Final = 60.0
DEFAULT_AGENT_TIMEOUT_SECONDS: Final = 1800.0
DEFAULT_TERMINATION_GRACE_SECONDS: Final = 10.0


class TimeoutResultRaw(TypedDict):
    runtime: str
    agent: str
    timed_out: bool
    completion_confirmed: bool
    trajectory_exported: bool


class TimeoutResult(TypedDict):
    status: str
    summary: str
    raw: TimeoutResultRaw


@dataclass(frozen=True, slots=True)
class SupervisorSettings:
    output_dir: Path
    source_trajectory: Path
    target_trajectory: Path
    timeout_seconds: float
    termination_grace_seconds: float = 10.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "SupervisorSettings":
        output_dir = Path(env.get("ARENA_OUTPUT_DIR", "/envarena/out"))
        attempt_dir = Path(env.get("ARENA_ATTEMPT_DIR", output_dir))
        source_trajectory = attempt_dir / "terminus2" / "trajectory.json"
        trajectory_path = env.get("ARENA_TRAJECTORY_PATH")
        target_trajectory = (
            Path(trajectory_path)
            if trajectory_path
            else output_dir / "trajectory.jsonl"
        )
        return cls(
            output_dir=output_dir,
            source_trajectory=source_trajectory,
            target_trajectory=target_trajectory,
            timeout_seconds=max(
                1.0,
                float(
                    env.get(
                        "ARENA_AGENT_TIMEOUT_SEC",
                        str(DEFAULT_AGENT_TIMEOUT_SECONDS),
                    )
                )
                - OUTER_DEADLINE_GUARD_SECONDS,
            ),
            termination_grace_seconds=DEFAULT_TERMINATION_GRACE_SECONDS,
        )


def _terminate_worker(
    process: subprocess.Popen[bytes],
    grace_seconds: float,
) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _export_partial_trajectory(settings: SupervisorSettings) -> bool:
    try:
        export_trajectory(settings.source_trajectory, settings.target_trajectory)
    except (OSError, UnicodeError, json.JSONDecodeError, IncompleteTrajectoryError):
        logging.warning(
            "Terminus-2 timed out without an exportable partial trajectory",
            exc_info=True,
        )
        return False
    return True


def _write_timeout_result(
    settings: SupervisorSettings,
    trajectory_exported: bool,
) -> None:
    result: TimeoutResult = {
        "status": "OK",
        "summary": (
            "Terminus-2 reached the Agent deadline; partial workspace is preserved "
            "and reward determines task correctness"
        ),
        "raw": {
            "runtime": "supervisor",
            "agent": "terminus-2",
            "timed_out": True,
            "completion_confirmed": False,
            "trajectory_exported": trajectory_exported,
        },
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "harness_result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


def run_supervisor(
    settings: SupervisorSettings,
    command: Sequence[str],
) -> int:
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=settings.timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_worker(process, settings.termination_grace_seconds)
        trajectory_exported = _export_partial_trajectory(settings)
        _write_timeout_result(settings, trajectory_exported)
        return 0


def main() -> int:
    settings = SupervisorSettings.from_env(os.environ)
    return run_supervisor(
        settings,
        (sys.executable, "-m", "envarena_terminus2.runner"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
