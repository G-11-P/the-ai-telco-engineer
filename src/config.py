# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union


PROJECT_ROOT = Path(__file__).parent.parent


CAPABILITY_PROFILES: dict[str, dict[str, Any]] = {
    "planetary_6g_advanced": {
        "agent_llm": {
            "temperature": 0.9,
            "top_p": 1.0,
            "model_kwargs": {"reasoning_effort": "medium"},
        },
        "manager_llm": {
            "temperature": 0.2,
            "top_p": 1.0,
            "model_kwargs": {"reasoning_effort": "high"},
        },
        "workspace": {
            "container": {
                "memory_limit": "32g",
                "pids_limit": 4096,
                "use_gpu": True,
            }
        },
        "hyperparameter_tuner": {
            "n_trials": 60,
            "timeout": 3600,
        },
        "tools_config": {
            "eval_timeout": 600,
            "sionna_doc_config": {
                "cache_dir_path": "api_doc_cache",
                "retrieve_k": 20,
                "rerank_top_n": 8,
                "planetary_6g_config": {
                    "enabled": True,
                    "retrieve_k": 20,
                    "rerank_top_n": 8,
                    "cache_dir_path": "api_doc_cache_planetary_6g",
                },
                "summarize_llm": {
                    "temperature": 0.2,
                    "top_p": 1.0,
                    "model_kwargs": {"reasoning_effort": "low"},
                },
            },
        },
        "num_workers": 16,
        "num_gpus": 16,
        "population_size": 32,
        "num_generations": 300,
        "num_ideas": 16,
        "timeout": 1800,
        "task_submit_delay": 0.0,
        "result_processing_concurrency": -1,
        "enable_prompt_refinement": True,
        "num_off_front_candidates": 16,
        "off_front_temperature": 0.35,
    }
}


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base and return base."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _apply_capability_profile(config_data: dict[str, Any]) -> dict[str, Any]:
    """Apply an optional capability profile while preserving explicit overrides."""
    profile_name = config_data.get("capability_profile")
    if not profile_name:
        return config_data

    if profile_name not in CAPABILITY_PROFILES:
        supported = ", ".join(sorted(CAPABILITY_PROFILES))
        raise ValueError(
            f"Unknown capability_profile '{profile_name}'. "
            f"Supported values: {supported}"
        )

    defaults = deepcopy(CAPABILITY_PROFILES[profile_name])
    return _deep_merge_dict(defaults, config_data)


@dataclass
class LLMConfig:
    """Configuration for creating an LLM instance."""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.7
    top_p: float = 0.95
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True, frozen=True)
class ContainerConfig:
    # Name of the Docker image to use for the container.
    docker_image: str
    # Path to the Dockerfile, in case the image needs to be built. Will be resolved relative
    # to the folder containing `config.json` (i.e., the task's folder).
    # If None, then the `docker_image` must be a pre-built image or a standard Docker image
    # that will be downloaded from Docker Hub, e.g. `python:3.12-slim`.
    dockerfile_path: str | None
    # Memory limit for the container. Can be specified as a string with units (e.g. "16g", "1024m").
    memory_limit: str = "16g"
    # Maximum number of processes in the container.
    pids_limit: int = 2048
    # Whether to enable GPU access. Falls back to CPU if NVIDIA runtime unavailable.
    use_gpu: bool = True
    # Directory inside the Docker container mapped to the workspace.
    workspace_mount_point: str = "/workspace"


@dataclass(kw_only=True, frozen=True)
class WorkspaceConfig:
    """Configuration for workspace Docker containers."""

    # Parent directory for workspace directories (on the host).
    base_path: str = "workspaces"
    # Dcker container configuration.
    container: ContainerConfig = field(
        default_factory=lambda: ContainerConfig(
            docker_image="python:3.12-slim",
            dockerfile_path=None
        )
    )


class ToolsConfig:
    """Dynamic configuration for tools parameters.

    This class wraps a dictionary and provides attribute-style access.
    Any parameters defined in the JSON's tools_config section are available.

    Example JSON:
        "tools_config": {
            "cache_dir_path": "api_doc_cache",
            "custom_param": "value"
        }

    Usage:
        config.tools_config.cache_dir_path  # "api_doc_cache"
        config.tools_config.custom_param    # "value"
        config.tools_config.get("missing", "default")  # "default"
    """

    def __init__(self, data: Optional[dict[str, Any]] = None):
        """Initialize with a dictionary of parameters.

        Args:
            data: Dictionary of tool configuration parameters.
        """
        self._data = data if data is not None else {}

    def __getattr__(self, name: str) -> Any:
        """Get a configuration value by attribute name."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        if name not in self._data:
            raise AttributeError(f"ToolsConfig has no parameter '{name}'")
        return self._data[name]

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default."""
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration."""
        return key in self._data

    def __repr__(self) -> str:
        return f"ToolsConfig({self._data})"

    def to_dict(self) -> dict[str, Any]:
        """Return the underlying dictionary."""
        return self._data.copy()


@dataclass
class HyperparameterTunerConfig:
    """Configuration for post-process hyperparameter tuning (Optuna)."""

    n_trials: int = 30
    timeout: int = 300


@dataclass(kw_only=True, frozen=True)
class LogConfig:
    """Configuration for logging."""
    logging_level: int = logging.INFO
    logging_format: str = "%(asctime)s %(name)s [%(levelname)s] %(message)s"


@dataclass
class Config:
    """Configuration for the agent manager and optimization run."""

    # LLM configuration: both required (use LLMConfig for each)
    capability_profile: str = "default"
    agent_llm: LLMConfig = field(default_factory=LLMConfig)   # Used by agents (workers)
    manager_llm: LLMConfig = field(default_factory=LLMConfig)  # Used by manager (ideas, summaries)

    # Workspace configuration
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    tools_config: ToolsConfig = field(default_factory=ToolsConfig)
    logging_config: LogConfig = field(default_factory=LogConfig)

    # Post-process hyperparameter tuning
    hyperparameter_tuner: HyperparameterTunerConfig = field(
        default_factory=HyperparameterTunerConfig
    )

    # Worker configuration
    num_workers: int = 10
    num_gpus: int = 1

    # Metric optimization direction
    higher_is_better: bool = False

    # Evaluation timeout (seconds per evaluation run)
    eval_timeout: int = 240

    # Optimization parameters
    population_size: int = 20
    num_generations: int = 5
    num_ideas: int = 3  # Number of distinct algorithmic approaches per generation
    timeout: int = 900  # seconds per agent
    task_submit_delay: float = 30.0  # seconds between task submissions
    result_processing_concurrency: int = -1  # -1 means fully parallel (one worker per candidate)

    # Off-front diversification: in addition to the Pareto-front entries, the
    # orchestrator is shown a sample of off-front candidates (one per cluster
    # not already on the front, restricted to verdict='yes' entries). Sampling
    # weights are a stable softmax over the metric gap to the best candidate,
    # normalised by the pool's spread so the temperature is scale-invariant.
    # Set num_off_front_candidates to 0 to disable.
    num_off_front_candidates: int = 10
    off_front_temperature: float = 0.5

    # Prompt file
    prompt_path: str = "prompt.md"

    # Prompt refinement: when enabled, the orchestrator analyzes agent
    # journals after each generation and refines the agent prompt template.
    enable_prompt_refinement: bool = False

def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Config instance with the loaded configuration.
    """
    import printer

    # TODO: most of this logic could be automated by using OmegaConf.

    config_path = Path(config_path)
    with open(config_path) as f:
        # Remove comments (lines starting with //) for JSON parsing
        lines = f.readlines()
        clean_lines = [line for line in lines if not line.strip().startswith("//")]
        data = json.loads("".join(clean_lines))
    data = _apply_capability_profile(data)

    # Get API key from environment variable (required)
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        printer.log("Error: MODEL_API_KEY environment variable is not set.",
                    "Please set it with: export MODEL_API_KEY=<your-api-key>",
                    level=logging.ERROR)
        sys.exit(1)

    # Extract agent LLM config (required)
    if "agent_llm" not in data:
        printer.log("Error: config must contain 'agent_llm' section (LLM used by agents).")
        sys.exit(1)
    agent_llm_section = data["agent_llm"]
    _llm_defaults = LLMConfig()
    agent_llm = LLMConfig(
        api_key=api_key,
        base_url=agent_llm_section.get("base_url", _llm_defaults.base_url),
        model=agent_llm_section.get("model", _llm_defaults.model),
        temperature=agent_llm_section.get("temperature", _llm_defaults.temperature),
        top_p=agent_llm_section.get("top_p", _llm_defaults.top_p),
        model_kwargs=agent_llm_section.get("model_kwargs", {}),
    )

    # Extract manager LLM config (required)
    if "manager_llm" not in data:
        printer.log("Error: config must contain 'manager_llm' section (LLM used for ideas and summaries).")
        sys.exit(1)
    manager_llm_section = data["manager_llm"]
    manager_llm = LLMConfig(
        api_key=api_key,
        base_url=manager_llm_section.get("base_url", _llm_defaults.base_url),
        model=manager_llm_section.get("model", _llm_defaults.model),
        temperature=manager_llm_section.get("temperature", 0.0),
        top_p=manager_llm_section.get("top_p", _llm_defaults.top_p),
        model_kwargs=manager_llm_section.get("model_kwargs", {}),
    )

    # Extract workspace config from nested section
    workspace_section = deepcopy(data.get("workspace", {}))
    if "base_path" in workspace_section:
        workspace_section["base_path"] = str(config_path.parent / workspace_section["base_path"])

    container_section = workspace_section.get("container", {})
    if "dockerfile_path" in container_section:
        container_section["dockerfile_path"] = str(config_path.parent / container_section["dockerfile_path"])
    workspace_section["container"] = ContainerConfig(**container_section)

    workspace = WorkspaceConfig(**workspace_section)

    # Extract tools config from nested section (dynamic - any keys allowed)
    tools_section = data.get("tools_config", {})
    tools_config = ToolsConfig(tools_section)

    # Logging config
    logging_section = data.get("logging", {})
    logging_config = LogConfig(**logging_section)

    # eval_timeout: accept from tools_config (legacy) or top-level
    eval_timeout = tools_section.get(
        "eval_timeout",
        data.get("eval_timeout", Config.eval_timeout),
    )

    # Extract hyperparameter tuner config (top-level, not a tool)
    hp_section = data.get("hyperparameter_tuner", {})
    hp_tuner_config = HyperparameterTunerConfig(
        n_trials=hp_section.get("n_trials", HyperparameterTunerConfig.n_trials),
        timeout=hp_section.get("timeout", HyperparameterTunerConfig.timeout),
    )

    return Config(
        capability_profile=data.get("capability_profile", Config.capability_profile),
        agent_llm=agent_llm,
        manager_llm=manager_llm,
        workspace=workspace,
        tools_config=tools_config,
        logging_config=logging_config,
        hyperparameter_tuner=hp_tuner_config,
        num_workers=data.get("num_workers", Config.num_workers),
        num_gpus=data.get("num_gpus", Config.num_gpus),
        higher_is_better=data.get("higher_is_better", Config.higher_is_better),
        eval_timeout=eval_timeout,
        population_size=data.get("population_size", Config.population_size),
        num_generations=data.get("num_generations", Config.num_generations),
        num_ideas=data.get("num_ideas", Config.num_ideas),
        timeout=data.get("timeout", Config.timeout),
        task_submit_delay=data.get("task_submit_delay", Config.task_submit_delay),
        result_processing_concurrency=data.get(
            "result_processing_concurrency",
            Config.result_processing_concurrency,
        ),
        num_off_front_candidates=data.get(
            "num_off_front_candidates", Config.num_off_front_candidates
        ),
        off_front_temperature=data.get(
            "off_front_temperature", Config.off_front_temperature
        ),
        prompt_path=data.get("prompt_path", Config.prompt_path),
        enable_prompt_refinement=data.get("enable_prompt_refinement", Config.enable_prompt_refinement),
    )
