import json
from pathlib import Path

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class IncompleteTrajectoryError(RuntimeError):
    pass


def _load_trajectory(path: Path) -> dict[str, JsonValue]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise IncompleteTrajectoryError("trajectory root is not an object")
    return loaded


def trajectory_has_confirmed_completion(path: Path) -> bool:
    return trajectory_completion_count(path) >= 2


def trajectory_completion_count(path: Path) -> int:
    document = _load_trajectory(path)
    steps = document.get("steps")
    if not isinstance(steps, list):
        return 0
    completion_count = 0
    for step in reversed(steps):
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        calls = step.get("tool_calls")
        if not isinstance(calls, list):
            break
        completed = any(
            isinstance(call, dict) and call.get("function_name") == "mark_task_complete"
            for call in calls
        )
        if not completed:
            break
        completion_count += 1
    return completion_count


def export_trajectory(source: Path, target: Path) -> None:
    document = _load_trajectory(source)
    session_id = document.get("session_id")
    steps = document.get("steps")
    if not isinstance(steps, list):
        raise IncompleteTrajectoryError("trajectory has no steps")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as output:
        for step in steps:
            if isinstance(step, dict):
                output.write(json.dumps({"session_id": session_id, **step}) + "\n")
