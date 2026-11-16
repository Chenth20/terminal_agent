import logging
import os
from collections.abc import Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NotRequired, Protocol, TypedDict, cast

import anyio
from pydantic import BaseModel, ConfigDict
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential

from envarena_terminus2.settings import (
    RunnerConfigurationError as RunnerConfigurationError,
)
from envarena_terminus2.settings import RunnerSettings as RunnerSettings
from envarena_terminus2.settings import ThinkingOption as ThinkingOption
from envarena_terminus2.trajectory import (
    IncompleteTrajectoryError as IncompleteTrajectoryError,
)
from envarena_terminus2.trajectory import JsonScalar as JsonScalar
from envarena_terminus2.trajectory import JsonValue as JsonValue
from envarena_terminus2.trajectory import (
    export_trajectory,
    trajectory_completion_count,
    trajectory_has_confirmed_completion,
)

if TYPE_CHECKING:
    from harbor.models.agent.context import AgentContext

SESSION_NAME: Final = "terminus-2"
# Harness agent.env entries named TERMINUS_SANDBOX_ENV_<NAME>=<value> are exported as
# <NAME>=<value> inside the task tmux shell (pip/uv mirrors, timeouts, ...). They only
# reach the agent's shell, not the Reward stage, so the official test.sh is unaffected.
SANDBOX_ENV_PREFIX: Final = "TERMINUS_SANDBOX_ENV_"


def sandbox_extra_env(env: Mapping[str, str]) -> dict[str, str]:
    return {
        key[len(SANDBOX_ENV_PREFIX) :]: value
        for key, value in env.items()
        if key.startswith(SANDBOX_ENV_PREFIX) and len(key) > len(SANDBOX_ENV_PREFIX)
    }


class HarnessResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    summary: str
    raw: dict[str, str | int | float | bool | None]


class ChatTemplateKwargs(TypedDict):
    thinking_option: ThinkingOption


class InferenceExtraBody(TypedDict):
    top_k: int
    separate_reasoning: bool
    chat_template_kwargs: NotRequired[ChatTemplateKwargs]


class AuthorizationHeaders(TypedDict):
    Authorization: str


class JsonResponseFormat(TypedDict):
    type: Literal["json_object"]


class LLMCallKwargs(TypedDict):
    top_p: float
    num_retries: int
    extra_body: InferenceExtraBody
    extra_headers: AuthorizationHeaders
    response_format: NotRequired[JsonResponseFormat]


class RetryWrappedCall(Protocol):
    retry: AsyncRetrying


def build_llm_call_kwargs(settings: RunnerSettings) -> LLMCallKwargs:
    kwargs: LLMCallKwargs = {
        "top_p": settings.top_p,
        "num_retries": 0,
        "extra_body": {
            "top_k": settings.top_k,
            "separate_reasoning": settings.separate_reasoning,
        },
        "extra_headers": {
            "Authorization": f"Bearer {settings.api_key}",
        },
    }
    if settings.force_json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if settings.thinking_option is not None:
        kwargs["extra_body"]["chat_template_kwargs"] = {
            "thinking_option": settings.thinking_option
        }
    return kwargs


def build_model_info(settings: RunnerSettings) -> dict[str, int]:
    return {"max_input_tokens": settings.model_context_limit}


def configure_gateway_retries(settings: RunnerSettings) -> AsyncRetrying:
    from harbor.llms.lite_llm import LiteLLM

    retrying = cast(RetryWrappedCall, LiteLLM.call).retry
    retrying.stop = stop_after_attempt(settings.gateway_retry_attempts)
    retrying.wait = wait_random_exponential(multiplier=2, min=5, max=30)
    return retrying


async def _run_agent(settings: RunnerSettings) -> tuple["AgentContext", Path]:
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.models.agent.context import AgentContext

    from envarena_terminus2.environment import LocalEnvironment

    await anyio.Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    log_root = settings.attempt_dir or settings.output_dir
    logs_dir = log_root / "terminus2"
    await anyio.Path(logs_dir).mkdir(parents=True, exist_ok=True)
    await anyio.Path("/logs/agent").mkdir(parents=True, exist_ok=True)
    os.chdir(settings.workdir)
    os.environ["OPENAI_API_KEY"] = settings.api_key
    os.environ["ANTHROPIC_API_KEY"] = settings.api_key
    configure_gateway_retries(settings)

    environment = LocalEnvironment(
        session_id=os.environ.get("ARENA_TASK_ID", "arena-task")
    )
    await environment.exec(f"tmux kill-session -t {SESSION_NAME} 2>/dev/null || true")
    agent = Terminus2(
        logs_dir=logs_dir,
        model_name=settings.model_name,
        parser_name="json",
        api_base=settings.api_base,
        temperature=settings.temperature,
        reasoning_effort=settings.reasoning_effort,
        model_info=build_model_info(settings),
        max_turns=settings.max_turns,
        suppress_max_turns_warning=True,
        enable_summarize=True,
        proactive_summarization_threshold=(settings.proactive_summarization_threshold),
        record_terminal_session=False,
        interleaved_thinking=settings.interleaved_thinking,
        llm_call_kwargs=dict(build_llm_call_kwargs(settings)),
        extra_env={
            **sandbox_extra_env(os.environ),
            **{
                key: os.environ[f"TERMINUS_TASK_{key}"]
                for key in ("PATH", "LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH")
            },
        },
    )
    context = AgentContext()
    with anyio.fail_after(settings.agent_timeout_seconds):
        await agent.setup(environment)
        await agent.run(settings.query_path.read_text(), environment, context)
    return context, logs_dir / "trajectory.json"


def _write_result(settings: RunnerSettings, result: HarnessResult) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "harness_result.json").write_text(
        result.model_dump_json(indent=2) + "\n"
    )


def _export_partial_trajectory(settings: RunnerSettings) -> bool:
    log_root = settings.attempt_dir or settings.output_dir
    source = log_root / "terminus2" / "trajectory.json"
    if not source.is_file():
        return False
    try:
        export_trajectory(
            source,
            settings.trajectory_path or settings.output_dir / "trajectory.jsonl",
        )
    except (IncompleteTrajectoryError, JSONDecodeError, OSError, UnicodeError):
        logging.exception("Terminus-2 partial trajectory export failed")
        return False
    return True


def main() -> int:
    settings: RunnerSettings | None = None
    try:
        settings = RunnerSettings.from_env(os.environ)
        context, trajectory = anyio.run(_run_agent, settings)
        completion_count = trajectory_completion_count(trajectory)
        completion_confirmed = trajectory_has_confirmed_completion(trajectory)
        export_trajectory(
            trajectory,
            settings.trajectory_path or settings.output_dir / "trajectory.jsonl",
        )
        _write_result(
            settings,
            HarnessResult(
                status="OK",
                summary=(
                    "Terminus-2 completed with two task_complete confirmations"
                    if completion_confirmed
                    else "Terminus-2 run ended without double completion confirmation; "
                    "reward determines task correctness"
                ),
                raw={
                    "runtime": "runner",
                    "agent": "terminus-2",
                    "agent_version": "2.0.0",
                    "parser": "json",
                    "completion_confirmations": completion_count,
                    "completion_confirmed": completion_confirmed,
                    "input_tokens": context.n_input_tokens,
                    "output_tokens": context.n_output_tokens,
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "top_k": settings.top_k,
                    "separate_reasoning": settings.separate_reasoning,
                    "reasoning_effort": settings.reasoning_effort,
                    "model_context_limit": settings.model_context_limit,
                    "proactive_summarization_threshold": (
                        settings.proactive_summarization_threshold
                    ),
                    "max_turns": settings.max_turns,
                    "interleaved_thinking": settings.interleaved_thinking,
                    "tool_call_mode": settings.tool_call_mode,
                    "gateway_retry_attempts": settings.gateway_retry_attempts,
                    "force_json_object": settings.force_json_object,
                    "sandbox_env_keys": ",".join(sorted(sandbox_extra_env(os.environ))),
                },
            ),
        )
        return 0
    except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
        if settings is not None:
            partial_trajectory_exported = _export_partial_trajectory(settings)
            _write_result(
                settings,
                HarnessResult(
                    status="ERROR",
                    summary=f"{type(error).__name__}: {error}",
                    raw={
                        "runtime": "runner",
                        "agent": "terminus-2",
                        "partial_trajectory_exported": partial_trajectory_exported,
                    },
                ),
            )
        logging.exception("Terminus-2 runner failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
