from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

type ReasoningEffort = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "default"
]
type ThinkingOption = Literal["on", "off", "auto"]

THINKING_OPTION_ADAPTER: Final = TypeAdapter(ThinkingOption)

type ToolCallMode = Literal["none", "function"]
TOOL_CALL_MODE_ADAPTER: Final = TypeAdapter(ToolCallMode)


class RunnerConfigurationError(RuntimeError):
    pass


class RunnerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str
    api_base: str
    model_name: str
    query_path: Path
    output_dir: Path
    attempt_dir: Path | None = None
    trajectory_path: Path | None = None
    workdir: Path = Path("/app")
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    separate_reasoning: bool = True
    reasoning_effort: ReasoningEffort = "max"
    model_context_limit: int = 262_144
    proactive_summarization_threshold: int = 65_536
    agent_timeout_seconds: float = 1770.0
    max_turns: int | None = None
    gateway_retry_attempts: int = 6
    force_json_object: bool = False
    thinking_option: ThinkingOption | None = None
    interleaved_thinking: bool = True
    tool_call_mode: ToolCallMode = "none"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "RunnerSettings":
        generic_api_key = _first(env, "LLM_API_KEY", "OPENAI_API_KEY")
        api_key = generic_api_key or _first(
            env,
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
        )
        generic_base = _first(env, "LLM_BASE_URL", "OPENAI_BASE_URL")
        raw_base = generic_base or _first(env, "ANTHROPIC_BASE_URL")
        generic_model = _first(env, "LLM_MODEL", "OPENAI_MODEL")
        raw_model = generic_model or _first(env, "ANTHROPIC_MODEL")
        query_path = _first(env, "ARENA_QUERY_PATH")
        output_dir = _first(env, "ARENA_OUTPUT_DIR")
        required = {
            "LLM_API_KEY": api_key,
            "LLM_BASE_URL": raw_base,
            "LLM_MODEL": raw_model,
            "ARENA_QUERY_PATH": query_path,
            "ARENA_OUTPUT_DIR": output_dir,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RunnerConfigurationError(
                f"missing required settings: {', '.join(missing)}"
            )

        api_base = raw_base.rstrip("/")
        model_name = raw_model
        if "/" not in model_name:
            provider = "openai" if generic_model else "anthropic"
            model_name = f"{provider}/{model_name}"
        return cls(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            query_path=Path(query_path),
            output_dir=Path(output_dir),
            attempt_dir=_optional_path(env.get("ARENA_ATTEMPT_DIR")),
            trajectory_path=_optional_path(env.get("ARENA_TRAJECTORY_PATH")),
            workdir=Path(
                _first(
                    env,
                    "TERMINUS_WORKDIR",
                    "ARENA_AGENT_WORKDIR",
                    "ARENA_WORKSPACE",
                )
                or "/app"
            ),
            temperature=float(env.get("TERMINUS_TEMPERATURE", "1.0")),
            top_p=float(env.get("TERMINUS_TOP_P", "0.95")),
            top_k=int(env.get("TERMINUS_TOP_K", "50")),
            separate_reasoning=_parse_bool(
                env.get("TERMINUS_SEPARATE_REASONING", "true")
            ),
            reasoning_effort=cast(
                ReasoningEffort, env.get("TERMINUS_REASONING_EFFORT", "max")
            ),
            model_context_limit=int(env.get("TERMINUS_MODEL_CONTEXT_LIMIT", "262144")),
            proactive_summarization_threshold=int(
                env.get("TERMINUS_PROACTIVE_SUMMARIZATION_THRESHOLD", "65536")
            ),
            agent_timeout_seconds=max(
                1.0,
                float(env.get("ARENA_AGENT_TIMEOUT_SEC", "1800")) - 30.0,
            ),
            max_turns=_optional_int(env.get("TERMINUS_MAX_TURNS")),
            gateway_retry_attempts=int(env.get("TERMINUS_GATEWAY_RETRY_ATTEMPTS", "6")),
            force_json_object=_parse_bool(
                env.get("TERMINUS_FORCE_JSON_OBJECT", "false")
            ),
            thinking_option=_optional_thinking_option(
                env.get("TERMINUS_THINKING_OPTION")
            ),
            interleaved_thinking=_parse_bool(
                env.get("TERMINUS_INTERLEAVED_THINKING", "true")
            ),
            tool_call_mode=_optional_tool_call_mode(
                env.get("TERMINUS_TOOL_CALL_MODE")
            ),
        )


def _first(env: Mapping[str, str], *names: str) -> str:
    return next((env[name] for name in names if env.get(name)), "")


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _optional_int(value: str | None) -> int | None:
    return int(value) if value else None


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_thinking_option(value: str | None) -> ThinkingOption | None:
    if not value:
        return None
    return THINKING_OPTION_ADAPTER.validate_python(value.strip().lower())

def _optional_tool_call_mode(value: str | None) -> ToolCallMode:
    return TOOL_CALL_MODE_ADAPTER.validate_python(value) if value else "none"
