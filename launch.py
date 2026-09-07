# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Launch script for optimization agents.

Takes a task folder path as argument. The task folder must contain:
- config.json: Configuration file for the agent orchestrator
- eval_tool.py: Evaluation tool module with an `EvalTool` class
- Any other files required to perform the optimization

Usage:
    python launch.py <task_folder>

Example:
    python launch.py tasks/chn_est
"""

import argparse
import importlib.util
from pathlib import Path
import sys

# Add src/ to path before importing from it
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config
from orchestrator import AgentOrchestrator
import printer


def load_eval_tool(task_folder: Path):
    """Dynamically load the EvalTool class from the task folder.

    Args:
        task_folder: Path to the task folder containing eval_tool.py

    Returns:
        The EvalTool class from the task's eval_tool.py module.
    """
    eval_tool_path = task_folder / "eval_tool.py"
    if not eval_tool_path.exists():
        raise FileNotFoundError(f"eval_tool.py not found in {task_folder}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("eval_tool", eval_tool_path)
    module = importlib.util.module_from_spec(spec)

    # Add task folder to path so eval_tool can import local modules
    sys.path.insert(0, str(task_folder))

    # Register module in sys.modules so pickle can find it
    sys.modules["eval_tool"] = module

    spec.loader.exec_module(module)

    # Get the EvalTool class
    if not hasattr(module, "EvalTool"):
        raise AttributeError(f"eval_tool.py must define an 'EvalTool' class")

    return module.EvalTool


def load_tool_factory(task_folder: Path):
    """Optionally load the ToolFactory class from the task folder.

    Args:
        task_folder: Path to the task folder potentially containing tool_factory.py

    Returns:
        The ToolFactory class if found, None otherwise.
    """
    tool_factory_path = task_folder / "tool_factory.py"
    if not tool_factory_path.exists():
        return None

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("tool_factory", tool_factory_path)
    module = importlib.util.module_from_spec(spec)

    # Register module in sys.modules so pickle can find it
    sys.modules["tool_factory"] = module

    spec.loader.exec_module(module)

    # Get the ToolFactory class if it exists
    if not hasattr(module, "ToolFactory"):
        printer.warning("Warning: tool_factory.py exists but doesn't define 'ToolFactory' class")
        return None

    return module.ToolFactory


def main():
    parser = argparse.ArgumentParser(
        description="Launch optimization agents for a task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python launch.py tasks/chn_est
    python launch.py tasks/link_adaptation
        """
    )

    parser.add_argument(
        "task_folder",
        type=str,
        help="Path to the task folder containing config.json and eval_tool.py"
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help=(
            "Validate task config/module wiring and exit without launching workers."
        ),
    )
    args = parser.parse_args()

    # Resolve task folder path
    task_folder = Path(args.task_folder).resolve()
    if not task_folder.exists():
        printer.error(f"Error: Task folder not found: {task_folder}")
        sys.exit(1)

    # Load configuration
    config_path = task_folder / "config.json"
    if not config_path.exists():
        printer.error(f"Error: config.json not found in {task_folder}")
        sys.exit(1)
    try:
        config = load_config(config_path)
    except Exception as e:
        printer.error(f"Error while loading config: {e}")
        sys.exit(1)

    printer.init(config.logging_config, None, "LAUNCHER")

    # Load the evaluation tool (required)
    try:
        EvalTool = load_eval_tool(task_folder)
    except (FileNotFoundError, AttributeError) as e:
        printer.error(f"Error: {e}")
        sys.exit(1)

    # Load the tool factory (optional)
    ToolFactory = load_tool_factory(task_folder)

    printer.info(f"Task folder: {task_folder}")
    printer.info(f"Config: {config_path}")
    printer.info(f"Prompt: {task_folder / config.prompt_path}")
    printer.info(f"Eval tool: {EvalTool.__name__}")
    if ToolFactory:
        printer.info(f"Tool factory: {ToolFactory.__name__}")

    if args.validate_config_only:
        printer.info("Validation successful. Exiting without launching workers.")
        return

    # Create and run agent orchestrator
    orchestrator = AgentOrchestrator(
        config,
        evaluation_tool_type=EvalTool,
        task_folder=task_folder,
        tool_factory_type=ToolFactory,
    )

    with orchestrator:
        orchestrator.run()


if __name__ == "__main__":
    main()
