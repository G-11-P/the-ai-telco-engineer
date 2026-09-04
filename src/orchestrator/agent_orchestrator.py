# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""AgentOrchestrator — top-level driver for idea-driven optimization.

Composes :class:`WorkerPool`, :class:`OrchestratorLLM`, and the workspace I/O
helpers to run a multi-generation optimisation loop.

Algorithm overview:
1. The orchestrator LLM produces *n* distinct algorithmic ideas from the task
   query (or from previous generation results).
2. A population of *m* agents is split across ideas (*m/n* per idea).
3. When a task completes the orchestrator LLM summarises the solution.
4. When a generation completes the orchestrator LLM produces *n* new ideas.

Solutions are evaluated on two objectives (metric and complexity).
The global Pareto front drives idea generation across generations.
"""

import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import fields
from pathlib import Path
from typing import Optional

import printer
from config import Config
from leaderboard import Candidate, ClusteredLeaderboard
from tool_lib.base import ToolProvider
from tool_lib.hp import bake_hyperparams

from .models import Idea, GenerationSummary, JournalAnalysis, Task, TaskResult
from .orchestrator_llm import OrchestratorLLM
from .prompt_engineer import PromptEngineer
from .worker_pool import WorkerPool
from .workspace_io import read_workspace_code, read_results


ProcessedResult = tuple[Candidate, Optional[JournalAnalysis]]

# Workspace IDs are minted as ``f"gen{generation:02d}-{counter:04d}"`` (see
# ``_submit_task``). Reference workspace IDs originate from the orchestrator LLM
# and are therefore untrusted (indirect prompt injection): validate them against
# this exact shape before using them as a host filesystem path component, so a
# value like ``"../../etc"`` can never traverse outside the workspace root.
_WORKSPACE_ID_RE = re.compile(r"^gen\d{2,}-\d{4,}$")


def resolve_num_result_processing_workers(
    configured: int,
    num_candidates: int,
) -> int:
    """Resolve configured result-processing concurrency for one generation."""
    if configured == -1:
        return num_candidates
    if configured < -1 or configured == 0:
        raise ValueError(
            "result_processing_concurrency must be -1 or a positive integer"
        )
    return min(configured, num_candidates)


class AgentOrchestrator:
    """Manages a pool of agents running as separate processes.

    Uses an orchestrator LLM for generating ideas and summarising solutions;
    candidates are clustered by idea.

    Can be used as a context manager::

        with AgentOrchestrator(config, EvalTool, task_folder) as orch:
            orch.run()
    """

    def __init__(
        self,
        config: Config,
        evaluation_tool_type: type[ToolProvider],
        task_folder: Path,
        tool_factory_type: Optional[type[ToolProvider]],
    ):
        self.config = config
        self._task_folder = Path(task_folder)
        self._workspace_root = Path(config.workspace.base_path)

        prompt_path = self._task_folder / config.prompt_path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        self._prompt = prompt_path.read_text()

        self._pool = WorkerPool(
            config=config,
            num_workers=config.num_workers,
            agent_llm=config.agent_llm,
            workspace_config=config.workspace,
            evaluation_tool_type=evaluation_tool_type,
            tool_factory_type=tool_factory_type,
            tools_config=config.tools_config,
            higher_is_better=config.higher_is_better,
            eval_timeout=config.eval_timeout,
            num_gpus=config.num_gpus,
            hp_tuner_config=config.hyperparameter_tuner,
        )
        printer.init(config.logging_config, self._pool.print_lock, "ORCHESTRATOR")

        log_lock = threading.Lock()
        self._llm = OrchestratorLLM(
            config.manager_llm,
            log_path=self._workspace_root / "orchestrator.log",
            higher_is_better=config.higher_is_better,
            log_lock=log_lock,
        )

        self._prompt_engineer = PromptEngineer(
            config.manager_llm,
            log_path=self._workspace_root / "orchestrator.log",
            higher_is_better=config.higher_is_better,
            log_lock=log_lock,
        )
        self._current_prompt_template: str = ""

        self._candidate_counter = 0
        self._workspace_to_idea: dict[str, tuple[int, str]] = {}

    @property
    def _leaderboard_path(self) -> Path:
        return self._workspace_root / "leaderboard.json"

    @property
    def _agent_prompt_path(self) -> Path:
        return self._workspace_root / "agent_prompt.md"

    def _next_candidate_counter(self, fallback: int) -> int:
        """Compute a candidate counter that won't collide with on-disk workspaces.

        Scans ``self._workspace_root`` for ``gen<NN>-<NNNN>`` directories
        (including any orphans from a mid-generation crash that never made
        it into ``leaderboard.json``) and returns ``max(<NNNN>) + 1``.
        Falls back to *fallback* when no matching directories exist.
        """
        if not self._workspace_root.exists():
            return fallback
        pattern = re.compile(r"^gen\d+-(\d+)$")
        max_idx = -1
        for entry in self._workspace_root.iterdir():
            if not entry.is_dir():
                continue
            m = pattern.match(entry.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        if max_idx < 0:
            return fallback
        return max_idx + 1

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        self._pool.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.stop()
        return False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> ClusteredLeaderboard:
        """Run the optimisation, resuming from an existing leaderboard if present.

        All parameters (population_size, num_generations, etc.) come from the
        :class:`Config` provided at initialisation.

        Returns:
            The :class:`ClusteredLeaderboard` with all candidates.
        """
        if self._agent_prompt_path.exists():
            self._current_prompt_template = self._agent_prompt_path.read_text()
            printer.log(
                f"Resuming: Loaded refined agent prompt template from "
                f"{self._agent_prompt_path}"
            )

        if self._leaderboard_path.exists():
            leaderboard = ClusteredLeaderboard.load(self._leaderboard_path)
            all_candidates = leaderboard.get_all_candidates()
            printer.log(
                f"Resuming: Loaded existing leaderboard with "
                f"{len(all_candidates)} candidates "
                f"in {len(leaderboard.clusters)} clusters"
            )
            if all_candidates:
                start_generation = max(c.generation for c in all_candidates) + 1
                self._candidate_counter = self._next_candidate_counter(
                    fallback=len(all_candidates),
                )
            else:
                start_generation = 0
                self._candidate_counter = self._next_candidate_counter(
                    fallback=0,
                )
            return self._run(
                query=self._prompt,
                leaderboard=leaderboard,
                start_generation=start_generation,
            )

        self._candidate_counter = self._next_candidate_counter(fallback=0)
        return self._run(query=self._prompt)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _run(
        self,
        query: str,
        leaderboard: Optional[ClusteredLeaderboard] = None,
        start_generation: int = 0,
    ) -> ClusteredLeaderboard:
        cfg = self.config
        is_continuation = leaderboard is not None and start_generation > 0

        self._print_header(query, leaderboard, start_generation, is_continuation)

        if leaderboard is None:
            leaderboard = ClusteredLeaderboard(
                query=query, higher_is_better=cfg.higher_is_better
            )
            leaderboard.save(self._leaderboard_path)

        for gen_offset in range(cfg.num_generations):
            generation = start_generation + gen_offset
            printer.section(
                "", "=" * 60, f"Generation {generation}", "=" * 60, ""
            )

            front = leaderboard.pareto_front()
            off_front = leaderboard.sample_off_front(
                k=cfg.num_off_front_candidates,
                temperature=cfg.off_front_temperature,
            )
            use_initial = generation == 0 or not front
            try:
                if use_initial:
                    descriptions = self._llm.generate_initial_ideas(
                        query, cfg.num_ideas
                    )
                    refs_per_idea: list[list[str]] = [[] for _ in descriptions]
                else:
                    summary_fields = {f.name for f in fields(GenerationSummary)}

                    def _to_summary(entry: dict) -> GenerationSummary:
                        return GenerationSummary(
                            **{
                                k: v
                                for k, v in entry.items()
                                if k in summary_fields
                            }
                        )

                    front_summaries = [_to_summary(e) for e in front]
                    off_front_summaries = [_to_summary(e) for e in off_front]
                    front_codes = [e.get("code", "") for e in front]
                    off_front_codes = [e.get("code", "") for e in off_front]
                    descriptions, refs_per_idea = (
                        self._llm.generate_ideas_from_results(
                            front_summaries,
                            off_front_summaries,
                            cfg.num_ideas,
                            query,
                            front_codes=front_codes,
                            off_front_codes=off_front_codes,
                        )
                    )
            except Exception as e:
                phase = (
                    "initial ideas" if use_initial
                    else "ideas from previous results"
                )
                printer.log(
                    f"ERROR: Orchestrator LLM failed to generate {phase}: {e}"
                )
                printer.log("Stopping optimization.")
                break

            ids = leaderboard.get_next_cluster_ids(len(descriptions))
            ideas = [
                Idea(cid, desc, refs)
                for cid, desc, refs in zip(ids, descriptions, refs_per_idea)
            ]

            idea_lines = [f"Ideas for this generation ({len(ideas)}):"]
            for idea in ideas:
                truncated = (
                    idea.description[:80] + "..."
                    if len(idea.description) > 80
                    else idea.description
                )
                idea_lines.append(f"  [{idea.cluster_id}] {truncated}")
            printer.section(*idea_lines)

            for idea in ideas:
                leaderboard.add_cluster(idea.cluster_id, idea.description)

            try:
                journal_analyses = self._run_generation(
                    query=query,
                    ideas=ideas,
                    population_size=cfg.population_size,
                    generation=generation,
                    timeout=cfg.timeout,
                    leaderboard=leaderboard,
                    task_submit_delay=cfg.task_submit_delay,
                )
            except Exception as e:
                printer.log(f"ERROR: Generation {generation} failed: {e}")
                printer.log("Stopping optimization.")
                break

            self._print_generation_summary(leaderboard, generation)

            if cfg.enable_prompt_refinement and journal_analyses:
                try:
                    from agent import RESULT_PROMPT_TEMPLATE
                    base = (
                        self._current_prompt_template
                        or RESULT_PROMPT_TEMPLATE
                    )
                    refined = self._prompt_engineer.refine_prompt(
                        base, journal_analyses, generation,
                    )
                    self._current_prompt_template = refined
                    self._agent_prompt_path.write_text(refined)
                    printer.log(
                        f"Prompt template refined for next generation "
                        f"(saved to {self._agent_prompt_path})."
                    )
                except Exception as e:
                    printer.log(
                        f"WARNING: Prompt refinement failed: {e} "
                        "— keeping previous template."
                    )

        self._print_final_summary(leaderboard)
        return leaderboard

    # ------------------------------------------------------------------
    # Single generation
    # ------------------------------------------------------------------

    def _run_generation(
        self,
        query: str,
        ideas: list[Idea],
        population_size: int,
        generation: int,
        timeout: int,
        leaderboard: ClusteredLeaderboard,
        task_submit_delay: float,
    ) -> list[JournalAnalysis]:
        n = len(ideas)
        tasks_per_idea = max(1, population_size // n)
        num_candidates = tasks_per_idea * n
        self._workspace_to_idea.clear()

        printer.log(
            f"Submitting {num_candidates} tasks "
            f"({tasks_per_idea} per idea, {task_submit_delay}s stagger)..."
        )

        submitted_ids: list[str] = []
        for idx in range(num_candidates):
            idea = ideas[idx % n]
            workspace_id = self._submit_task(
                query=query,
                idea=idea,
                timeout=timeout,
                generation=generation,
            )
            self._workspace_to_idea[workspace_id] = (
                idea.cluster_id,
                idea.description,
            )
            submitted_ids.append(workspace_id)
            if idx < num_candidates - 1 and task_submit_delay > 0:
                time.sleep(task_submit_delay)

        printer.log(
            f"Submitted {len(submitted_ids)} tasks. Waiting for results..."
        )

        result_workers = resolve_num_result_processing_workers(
            self.config.result_processing_concurrency,
            num_candidates,
        )
        printer.log(
            f"Processing completed results with up to "
            f"{result_workers} concurrent jobs."
        )

        completed = 0
        received = 0
        journal_analyses: list[JournalAnalysis] = []
        pending: dict[Future[ProcessedResult], TaskResult] = {}

        with ThreadPoolExecutor(max_workers=result_workers) as executor:
            while received < num_candidates or pending:
                # Not all candidates have finished running
                if received < num_candidates:
                    result = self._pool.get_result(
                        timeout=0.1 if pending else None
                    )
                    if result is not None:
                        received += 1
                        future = executor.submit(self._process_result, result)
                        pending[future] = result
                        continue

                # No results are ready but not yet processed
                if not pending:
                    continue

                processed = {future for future in pending if future.done()}
                if not processed and received >= num_candidates:
                    processed, _ = wait(pending, return_when=FIRST_COMPLETED)

                # No new processsed candidates
                if not processed:
                    continue

                # Log processed candidates and add them to th leaderboard
                for future in processed:
                    result = pending.pop(future)
                    try:
                        candidate, analysis = future.result()
                    except Exception as exc:
                        completed += 1
                        printer.log(
                            f"[{completed}/{num_candidates}] ✗ "
                            f"{result.workspace_id} | processing error: {exc}"
                        )
                        continue
                    leaderboard.add_candidate(candidate)
                    leaderboard.save(self._leaderboard_path)
                    if analysis is not None:
                        journal_analyses.append(analysis)

                    completed += 1
                    status = "✓" if candidate.success else "✗"
                    n_hp = (
                        len(candidate.hp_results)
                        if candidate.success
                        else 0
                    )
                    hp_str = f" ({n_hp} HP configs)" if n_hp > 0 else ""
                    cluster_str = (
                        f"[{candidate.cluster}]"
                        if candidate.cluster is not None
                        else ""
                    )
                    error_str = ""
                    if not candidate.success and candidate.error:
                        error_str = f" | error: {candidate.error}"
                    printer.log(
                        f"[{completed}/{num_candidates}] {status} "
                        f"{result.workspace_id} {cluster_str}"
                        f"{hp_str}{error_str}"
                    )

        return journal_analyses

    # ------------------------------------------------------------------
    # Task building
    # ------------------------------------------------------------------

    def _submit_task(
        self,
        query: str,
        idea: Idea,
        timeout: Optional[int] = None,
        generation: int = 0,
    ) -> str:
        """Build a :class:`Task` for *idea* and submit it to the worker pool.

        Returns the workspace ID assigned to this task.
        """
        workspace_id = f"gen{generation:02d}-{self._candidate_counter:04d}"
        self._candidate_counter += 1

        approach = self._build_approach_section(idea)
        task = Task(
            workspace_id=workspace_id,
            query=query,
            assigned_approach_section=approach,
            prompt_template=self._current_prompt_template,
            timeout=timeout,
            generation=generation,
        )
        self._pool.submit(task)
        return workspace_id

    def _build_approach_section(self, idea: Idea) -> str:
        """Format the assigned-approach section that agents receive.

        Reference code from Pareto-front entries is baked with the
        specific HP configuration so agents see the actual defaults.
        The reference metric is intentionally not shown — it would
        otherwise act as an anchor target and bias the agent toward
        editing the reference rather than implementing the assigned
        approach.
        """
        lines = [
            "=== ASSIGNED APPROACH (you MUST follow this) ===",
            idea.description.strip(),
        ]
        for ref_entry in idea.reference_workspace_ids:
            ref_id, hp_index = self._parse_ref(ref_entry)
            if not ref_id:
                printer.warning(
                    f"Ignoring invalid reference workspace id: {ref_entry!r}"
                )
                continue
            ref_code = read_workspace_code(self._workspace_root, ref_id)
            success, hp_results = read_results(
                self._workspace_root, ref_id,
            )
            if not success or not ref_code or not hp_results:
                continue

            # Find the specific HP result
            matching = [h for h in hp_results if h.hp_index == hp_index]
            if not matching:
                matching = hp_results[:1]
            hp = matching[0]

            # Bake HP params into the code so the agent sees actual defaults
            if hp_index != 0 and hp.params:
                ref_code = bake_hyperparams(ref_code, hp.params)

            lines.append("")
            lines.append(
                f"Reference code from workspace {ref_id} "
                "(provided as a starting point — your job is to modify it "
                "to follow the assigned approach above, not to reproduce it):"
            )
            lines.append("```python")
            lines.append(ref_code.strip())
            lines.append("```")

        lines.append("")
        lines.append("=== END ASSIGNED APPROACH ===")
        return "\n".join(lines)

    @staticmethod
    def _parse_ref(ref_entry: str) -> tuple[str, int]:
        """Parse a reference string like ``"gen00-0001:2"`` into (workspace_id, hp_index).

        If no ``:hp_index`` suffix is present, defaults to 0.

        The workspace ID is validated against :data:`_WORKSPACE_ID_RE` because it
        originates from the orchestrator LLM and is later used as a
        host filesystem path component. An ID that does not match the expected
        ``genNN-NNNN`` shape (e.g. a ``../`` traversal payload) yields an empty
        ID so the caller skips it before touching the filesystem.
        """
        ref_id = ref_entry.strip()
        hp_index = 0
        if ":" in ref_entry:
            parts = ref_entry.rsplit(":", 1)
            try:
                ref_id, hp_index = parts[0].strip(), int(parts[1].strip())
            except ValueError:
                ref_id, hp_index = ref_entry.strip(), 0

        if not _WORKSPACE_ID_RE.match(ref_id):
            return "", 0
        return ref_id, hp_index

    # ------------------------------------------------------------------
    # Result processing
    # ------------------------------------------------------------------

    def _process_result(self, result: TaskResult) -> ProcessedResult:
        """Read a completed task result and build its candidate summary."""
        success, hp_results = read_results(
            self._workspace_root,
            result.workspace_id,
        )
        code = read_workspace_code(self._workspace_root, result.workspace_id)

        idea_id, idea_description = self._workspace_to_idea.get(
            result.workspace_id, (0, "Unknown idea")
        )

        summary = self._llm.summarize_solution(
            code, idea_description, workspace_id=result.workspace_id
        )

        analysis = None
        if self.config.enable_prompt_refinement:
            try:
                analysis = self._prompt_engineer.analyze_journal(
                    self._workspace_root, result.workspace_id,
                )
            except Exception as e:
                printer.log(
                    f"WARNING: Journal analysis failed for "
                    f"{result.workspace_id}: {e}"
                )

        if success and hp_results:
            candidate = Candidate(
                workspace_id=result.workspace_id,
                generation=result.generation,
                code=code,
                summary=summary,
                success=True,
                error=None,
                cluster=idea_id,
                hp_results=hp_results,
            )
        else:
            error_msg = "No result.json found or evaluation failed"
            if result.error:
                error_msg = f"{error_msg} ({result.error})"
            candidate = Candidate(
                workspace_id=result.workspace_id,
                generation=result.generation,
                code=code,
                summary=summary,
                success=False,
                error=error_msg,
                cluster=idea_id,
            )

        return candidate, analysis

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_header(
        self,
        query: str,
        leaderboard: Optional[ClusteredLeaderboard],
        start_generation: int,
        is_continuation: bool,
    ):
        cfg = self.config
        lines = [
            "=" * 60,
            (
                f"Continuing Optimization from Generation {start_generation}"
                if is_continuation
                else "Starting Idea-driven Optimization"
            ),
            "=" * 60,
            f"Query: {query[:100]}{'...' if len(query) > 100 else ''}",
        ]
        if is_continuation and leaderboard is not None:
            lines.append(
                f"Existing candidates: "
                f"{len(leaderboard.get_all_candidates())}"
            )
            lines.append(f"Existing clusters: {len(leaderboard.clusters)}")
        lines.extend(
            [
                f"Capability profile: {cfg.capability_profile}",
                f"Population size: {cfg.population_size}",
                f"Number of ideas: {cfg.num_ideas}",
                f"Generations: {cfg.num_generations}",
                f"Timeout per agent: {cfg.timeout}s",
                f"Task submit delay: {cfg.task_submit_delay}s",
                f"Result processing concurrency: {cfg.result_processing_concurrency}",
                "=" * 60,
                "",
            ]
        )
        printer.section(*lines)

    def _print_generation_summary(
        self, leaderboard: ClusteredLeaderboard, generation: int
    ):
        gen_candidates = leaderboard.get_current_generation_candidates(
            generation
        )
        successful = [c for c in gen_candidates if c.success and c.hp_results]
        clusters_this_gen = set(
            c.cluster for c in gen_candidates if c.cluster is not None
        )

        lines = [
            f"Generation {generation} Summary:",
            f"  Total candidates: {len(gen_candidates)}",
            f"  Successful: {len(successful)}",
            f"  Clusters used: {len(clusters_this_gen)}",
        ]
        printer.section(*lines)

    def _print_final_summary(self, leaderboard: ClusteredLeaderboard):
        all_candidates = leaderboard.get_all_candidates()
        successful_all = leaderboard.get_successful_candidates()

        lines = [
            "=" * 60,
            "Optimization Complete",
            "=" * 60,
            f"Total candidates evaluated: {len(all_candidates)}",
            f"Total successful: {len(successful_all)}",
            f"Total clusters: {len(leaderboard.clusters)}",
        ]

        pareto = leaderboard.pareto_front()
        if pareto:
            lines.append(f"  Pareto front size: {len(pareto)}")

        lines.extend(
            [
                f"Leaderboard saved to: {self._leaderboard_path}",
                "=" * 60,
                "",
            ]
        )
        printer.section(*lines)
