# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import Config, load_config
from leaderboard import Candidate, ClusteredLeaderboard
from orchestrator.agent_orchestrator import (
    AgentOrchestrator,
    resolve_num_result_processing_workers,
)
from orchestrator.models import Idea, JournalAnalysis, Task, TaskResult


class FakePool:
    def __init__(self):
        self._results: queue.Queue[TaskResult] = queue.Queue()
        self.submitted: list[Task] = []

    def submit(self, task: Task) -> None:
        self.submitted.append(task)
        self._results.put(
            TaskResult(
                workspace_id=task.workspace_id,
                query=task.query,
                response="",
                success=True,
                generation=task.generation,
            )
        )

    def get_result(self, timeout=None):
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None


@pytest.mark.parametrize(
    ("configured", "num_candidates", "expected"),
    [
        (-1, 4, 4),
        (1, 4, 1),
        (2, 4, 2),
        (8, 4, 4),
    ],
)
def test_resolve_num_result_processing_workers(configured, num_candidates, expected):
    assert (
        resolve_num_result_processing_workers(configured, num_candidates)
        == expected
    )


@pytest.mark.parametrize("configured", [0, -2])
def test_resolve_num_result_processing_workers_rejects_invalid_values(configured):
    with pytest.raises(ValueError, match="result_processing_concurrency"):
        resolve_num_result_processing_workers(configured, 4)


def test_config_default_result_processing_concurrency_is_fully_parallel():
    assert Config().result_processing_concurrency == -1


def test_load_config_reads_result_processing_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agent_llm": {"model": "agent-model", "base_url": "http://llm"},
                "manager_llm": {
                    "model": "manager-model",
                    "base_url": "http://llm",
                },
                "workspace": {
                    "container": {
                        "docker_image": "python:3.12-slim",
                        "dockerfile_path": "Dockerfile",
                    }
                },
                "result_processing_concurrency": 3,
            }
        )
    )

    cfg = load_config(config_path)

    assert cfg.result_processing_concurrency == 3


def test_load_config_applies_planetary_advanced_profile_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "capability_profile": "planetary_6g_advanced",
                "agent_llm": {
                    "model": "agent-model",
                    "base_url": "http://llm",
                },
                "manager_llm": {
                    "model": "manager-model",
                    "base_url": "http://llm",
                },
                "workspace": {
                    "container": {
                        "docker_image": "python:3.12-slim",
                        "dockerfile_path": "Dockerfile",
                    }
                },
            }
        )
    )

    cfg = load_config(config_path)

    assert cfg.capability_profile == "planetary_6g_advanced"
    assert cfg.num_workers == 16
    assert cfg.num_gpus == 16
    assert cfg.num_generations == 300
    assert cfg.enable_prompt_refinement is True
    assert cfg.result_processing_concurrency == -1
    assert cfg.tools_config.get("eval_timeout") == 600
    assert cfg.agent_llm.model_kwargs.get("reasoning_effort") == "medium"
    assert cfg.manager_llm.model_kwargs.get("reasoning_effort") == "high"


def test_load_config_rejects_unknown_capability_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "capability_profile": "unknown_profile",
                "agent_llm": {"model": "agent-model", "base_url": "http://llm"},
                "manager_llm": {"model": "manager-model", "base_url": "http://llm"},
                "workspace": {
                    "container": {
                        "docker_image": "python:3.12-slim",
                        "dockerfile_path": "Dockerfile",
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="Unknown capability_profile"):
        load_config(config_path)


def test_generation_post_processing_runs_in_parallel_and_saves_once(tmp_path):
    orch = object.__new__(AgentOrchestrator)
    orch.config = Config(result_processing_concurrency=-1)
    orch._pool = FakePool()
    orch._workspace_root = tmp_path
    orch._candidate_counter = 0
    orch._workspace_to_idea = {}
    orch._current_prompt_template = ""

    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def process_result(result: TaskResult):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with active_lock:
            active -= 1
        return (
            Candidate(
                workspace_id=result.workspace_id,
                generation=result.generation,
                cluster=0,
                success=True,
            ),
            JournalAnalysis(
                workspace_id=result.workspace_id,
                behavioral_summary="ok",
            ),
        )

    orch._process_result = process_result

    leaderboard = ClusteredLeaderboard(query="q")
    analyses = orch._run_generation(
        query="q",
        ideas=[Idea(0, "idea a"), Idea(1, "idea b")],
        population_size=4,
        generation=0,
        timeout=1,
        leaderboard=leaderboard,
        task_submit_delay=0.0,
    )

    candidates = leaderboard.get_all_candidates()
    assert len(candidates) == 4
    assert len(analyses) == 4
    assert max_active > 1

    saved = ClusteredLeaderboard.load(tmp_path / "leaderboard.json")
    assert len(saved.get_all_candidates()) == 4
    assert sorted(c.workspace_id for c in saved.get_all_candidates()) == sorted(
        c.workspace_id for c in candidates
    )
